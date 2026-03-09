import sys
import re

with open('B16FullIntegration.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define replacement
new_code = """        # T12: RSSM Forward über Sequenz → h_seq (B, L, d_model)
        h_seq = self.rssm.forward_sequence(z_seq, act_seq, goal_p)

        # ---------------------------------------------------------
        # 1. World Model Training
        # ---------------------------------------------------------
        l_next_z = 0.0
        l_pred_img = 0.0
        l_goal = 0.0

        for t in range(L):
            ctx = h_seq[:, t]            # (B, d_model)
            act = act_seq[:, t]          # (B, 6)

            l_goal += torch.clamp(1 - F.cosine_similarity(
                F.normalize(ctx[:, :LATENT_DIM], dim=-1), goal_p, dim=-1
            ), 0.0, 1.0).mean()

            pzn = self.rssm.predict_next_z(ctx, act)
            l_next_z += F.mse_loss(pzn, z_next_seq[:, t].detach())

            # Pred-Image: nobs für Step t (Indices: t, L+t, 2L+t, ...)
            pred_img_t = self.decoder(pzn)
            nobs_t     = nobs_flat[range(t, B * L, L)]
            l_pred_img += combined_recon_loss(pred_img_t, nobs_t, ssim_weight=0.3)

        l_goal   /= L
        l_next_z   /= L
        l_pred_img /= L

        # T14: Reward-Prädiktor – eigene Stichprobe aus Gemini-Labels im Buffer
        dev = obs_flat.device
        gem_n = min(self.replay.gemini_count, 8)
        if gem_n >= 2:
            rb = self.replay.sample(gem_n, require_gemini=True)
            if rb is not None:
                with torch.no_grad():
                    _, _, z_rb = self.encoder(rb["obs"].permute(0, 3, 1, 2))
                pred_r   = self.reward_head(
                    torch.cat([z_rb.detach(), rb["actions"]], dim=-1)
                ).squeeze(-1)
                l_reward = F.mse_loss(pred_r, rb["gemini_rewards"])
            else:
                l_reward = torch.tensor(0.0, device=dev)
        else:
            l_reward = torch.tensor(0.0, device=dev)

        # T13: Szenen-Beschreibungs-Loss – schwache Supervision aus Gemini-Labels
        if gem_n >= 2:
            rb_s = self.replay.sample(gem_n, require_gemini=True)
            if rb_s is not None and any(rb_s["gemini_labels"]):
                label_indices = torch.tensor(
                    [self._label_to_vocab_idx(lbl) for lbl in rb_s["gemini_labels"]],
                    dtype=torch.long, device=dev
                )
                with torch.no_grad():
                    _, _, z_s = self.encoder(rb_s["obs"].permute(0, 3, 1, 2))
                l_scene = F.cross_entropy(self.scene_head(z_s.detach()), label_indices)
            else:
                l_scene = torch.tensor(0.0, device=dev)
        else:
            l_scene = torch.tensor(0.0, device=dev)

        fe = (1.0  * l_recon +
              0.5  * l_pred_img +       # Nächst-Frame-Prediction im Bildraum
              self.beta * l_kl +
              0.1  * l_next_z +         # Hilfsziel Latent (reduziert, l_pred_img übernimmt)
              0.1  * l_goal +
              0.1  * l_reward +         # T14: Reward-Prädiktor
              0.1  * l_scene)           # T13: Szenen-Beschreibung

        self.optimizer.zero_grad()
        fe.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            list(self.rssm.parameters()) +
            list(self.goal_proj.parameters()) +
            list(self.reward_head.parameters()) +
            list(self.scene_head.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()

        # ---------------------------------------------------------
        # 2. Actor-Critic Training (Imagination)
        # ---------------------------------------------------------
        self.actor_optimizer.zero_grad()
        self.optimizer.zero_grad()

        h_detached = h_seq.detach()
        action_weights = torch.tensor(
            [1.0, 1.0, 0.3, 0.3, 0.3, 0.3], device=dev
        )
        l_imitation = torch.tensor(0.0, device=dev)
        l_sigma = torch.tensor(0.0, device=dev)
        l_cam = torch.tensor(0.0, device=dev)

        for t in range(L):
            ctx = h_detached[:, t]
            act = act_seq[:, t]
            pa, ps = self.action_head(ctx)
            
            l_imitation += (action_weights * (pa - act).pow(2)).mean()
            l_cam += pa[:, 2:4].pow(2).mean()
            
            with torch.no_grad():
                ae = (pa.detach() - act).abs()
            ss = torch.clamp(ps, min=1e-4)
            l_sigma += torch.mean(torch.log(ss) + ae / ss)

        l_imitation /= L
        l_sigma /= L
        l_cam /= L
        l_cam_center = l_cam

        gem_count = self.replay.gemini_count
        efe_blend = EFE_BLEND_MAX * min(gem_count / EFE_GEMINI_RAMP, 1.0)
        
        l_img_actor = torch.tensor(0.0, device=dev)
        l_value = torch.tensor(0.0, device=dev)

        if gem_count >= EFE_GEMINI_RAMP:
            H_imag = 5
            h_flat = h_detached.reshape(-1, D_MODEL)
            gp_imag = goal_p.unsqueeze(1).expand(-1, L, -1).reshape(-1, LATENT_DIM)
            
            h_i = h_flat
            rewards = []
            values = []
            
            for _ in range(H_imag):
                a_i, _ = self.action_head(h_i)
                z_next = self.rssm.predict_next_z(h_i, a_i)
                gru_input = torch.cat([z_next, a_i, gp_imag], dim=-1)
                h_i = self.rssm.gru(gru_input, h_i)
                
                r_i = self.reward_head(torch.cat([z_next, a_i], dim=-1)).squeeze(-1)
                v_i = self.value_head(h_i).squeeze(-1)
                
                rewards.append(r_i)
                values.append(v_i)
                
            rewards = torch.stack(rewards, dim=0) # (H_imag, N)
            values = torch.stack(values, dim=0)   # (H_imag, N)
            
            gamma = 0.95
            lam = 0.95
            
            returns = torch.zeros_like(values)
            last_val = values[-1]
            returns[-1] = rewards[-1] + gamma * last_val
            for t in reversed(range(H_imag - 1)):
                returns[t] = rewards[t] + gamma * ((1 - lam) * values[t+1] + lam * returns[t+1])
                
            l_value = F.mse_loss(values, returns.detach())
            l_img_actor = -returns.mean()

        l_actor_total = efe_blend * l_img_actor + (1 - efe_blend) * l_imitation
        l_ac_final = 0.2 * l_actor_total + 0.05 * l_sigma + 0.05 * l_cam
        if gem_count >= EFE_GEMINI_RAMP:
            l_ac_final += 0.1 * l_value
            
        l_ac_final.backward()
        
        # Gradient-Norm pro Modul (Monitoring)
        def _grad_norm(module):
            return sum(p.grad.norm().item()**2
                       for p in module.parameters() if p.grad is not None)**0.5

        grad_norms = {
            "encoder":     _grad_norm(self.encoder),
            "decoder":     _grad_norm(self.decoder),
            "rssm":        _grad_norm(self.rssm),
            "action_head": _grad_norm(self.action_head),
            "goal_proj":   _grad_norm(self.goal_proj),
            "reward_head": _grad_norm(self.reward_head),
            "scene_head":  _grad_norm(self.scene_head),
        }
        
        torch.nn.utils.clip_grad_norm_(
            list(self.action_head.parameters()) + list(self.value_head.parameters()),
            max_norm=1.0
        )
        self.actor_optimizer.step()
        self.scheduler.step(l_recon.detach())
        l_action = l_actor_total  # for logging"""

pattern = r'        # T12: RSSM Forward über Sequenz.*?self\.scheduler\.step\(l_recon\.detach\(\)\)'
content = re.sub(pattern, new_code, content, flags=re.DOTALL)

with open('B16FullIntegration.py', 'w', encoding='utf-8') as f:
    f.write(content)
