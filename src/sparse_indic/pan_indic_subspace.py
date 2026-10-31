"""Module 4: Latent Pan-Indic Phonetic Alignment & Factorized Subspace."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.sparse_indic.brahmic_fallback import BRAHMIC_BASES


def get_brahmic_offset(c: str) -> Optional[int]:
    """Return the 0x00..0x7F relative Brahmic offset of character c."""
    cp = ord(c)
    if 0x0900 <= cp <= 0x0D7F:
        return cp & 0x7F
    return None


class PanIndicSubspace(nn.Module):
    """Factorized Pan-Indic Shared Anchor Matrix & Language Projection:
    
    E_lang^(k) = A_pan_indic[M_k] * P_k + Delta_k
    """

    def __init__(
        self,
        num_anchors: int = 128,
        rank_r: int = 64,
        d_model: int = 1536,
        languages: Optional[List[str]] = None
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.rank_r = rank_r
        self.d_model = d_model
        self.languages = languages or list(BRAHMIC_BASES.keys())

        # Shared Pan-Indic Anchor Matrix A_pan_indic in R^(num_anchors x rank_r)
        self.A_pan_indic = nn.Parameter(torch.randn(num_anchors, rank_r))
        nn.init.orthogonal_(self.A_pan_indic)

        # Per-language low-rank projections P_k in R^(rank_r x d_model)
        self.projections = nn.ParameterDict()
        # Per-language sparse residuals Delta_k
        self.residuals = nn.ParameterDict()

        # Shared base projection to align cognate spaces
        base_proj = torch.randn(rank_r, d_model)
        nn.init.orthogonal_(base_proj)

        for lang in self.languages:
            # Initialize P_k close to shared base projection with language-specific variance
            # Scale 0.008 ensures cognate alignment >= 0.85 at initialization (DoD-4B)
            p_k = base_proj.clone() + 0.008 * torch.randn(rank_r, d_model)
            self.projections[lang] = nn.Parameter(p_k)

    def compute_token_embedding(
        self,
        token_str: str,
        lang_code: str
    ) -> torch.Tensor:
        """Compute embedding vector for a token using factorized Pan-Indic subspace."""
        proj = self.projections.get(lang_code)
        if proj is None:
            # Fallback projection
            proj = next(iter(self.projections.values()))

        # Determine anchor index from Brahmic offset
        if len(token_str) > 0 and get_brahmic_offset(token_str[0]) is not None:
            anchor_idx = get_brahmic_offset(token_str[0])
        else:
            anchor_idx = hash(token_str) % self.num_anchors

        anchor_vec = self.A_pan_indic[anchor_idx]  # shape: (rank_r,)
        emb = torch.matmul(anchor_vec, proj)  # shape: (d_model,)
        return emb

    def compute_cognate_cosine_similarity(
        self,
        cognate_char_offset: int = 0x15,  # 0x15 corresponds to consonant 'ka'
        eval_langs: Optional[List[str]] = None
    ) -> float:
        """Calculate mean pairwise cosine similarity of cognate tokens across scripts.
        
        Evaluates Devanagari, Bengali, Kannada, Tamil, Telugu at initialization (DoD-4B).
        """
        eval_langs = eval_langs or ["hin_Deva", "ben_Beng", "kan_Knda", "tam_Taml", "tel_Telu"]
        embeddings = []

        anchor_vec = self.A_pan_indic[cognate_char_offset]
        for lang in eval_langs:
            proj = self.projections[lang]
            emb = torch.matmul(anchor_vec, proj)
            emb_norm = F.normalize(emb, p=2, dim=-1)
            embeddings.append(emb_norm)

        cos_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = torch.dot(embeddings[i], embeddings[j]).item()
                cos_sims.append(sim)

        return float(sum(cos_sims) / max(1, len(cos_sims)))
