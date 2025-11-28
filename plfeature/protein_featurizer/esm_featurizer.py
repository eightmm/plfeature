"""
ESM Embedding Extractor for Protein Sequences.

Supports ESM3 and ESMC models for per-residue embedding extraction.
BOS/EOS tokens are stored separately for flexibility.

Usage:
    from plfeature.esm_featurizer import ESMFeaturizer

    # Single model
    featurizer = ESMFeaturizer(model_type="esmc", model_name="esmc_600m")
    result = featurizer.extract("MKTIIALSYIFCLVFA")
    embeddings = result['embeddings']  # [L, D]
    bos_token = result['bos_token']    # [D]
    eos_token = result['eos_token']    # [D]

    # Extract from PDB (uses sequence from structure)
    result = featurizer.extract_from_pdb("protein.pdb")
"""

import torch
import logging
from typing import Dict, Literal, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ESMFeaturizer:
    """Extract protein embeddings using ESM3 or ESMC models."""

    # Model embedding dimensions
    MODEL_DIMS = {
        # ESMC models
        "esmc_300m": 960,
        "esmc_600m": 1152,
        # ESM3 models
        "esm3-open": 1536,
    }

    def __init__(
        self,
        model_type: Literal["esm3", "esmc"] = "esmc",
        model_name: str = "esmc_600m",
        device: str = "cuda",
    ):
        """
        Initialize ESM embedding extractor.

        Args:
            model_type: "esm3" or "esmc"
            model_name: Model variant name
                ESMC: "esmc_600m" (1152-dim), "esmc_300m" (960-dim)
                ESM3: "esm3-open" (1536-dim)
            device: "cuda" or "cpu"
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.model = None

        self._load_model()

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension for current model."""
        return self.MODEL_DIMS.get(self.model_name, 1152)

    def _load_model(self):
        """Load ESM3 or ESMC model."""
        try:
            if self.model_type == "esm3":
                from esm.models.esm3 import ESM3
                logger.info(f"Loading ESM3 model: {self.model_name}")

                try:
                    self.model = ESM3.from_pretrained(self.model_name)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ESM3 model '{self.model_name}'.\n"
                        f"Available models: esm3-open\n"
                        f"Error: {e}"
                    )

                if self.device == "cpu":
                    self.model = self.model.float()
                self.model = self.model.to(self.device)

            elif self.model_type == "esmc":
                from esm.models.esmc import ESMC
                logger.info(f"Loading ESMC model: {self.model_name}")

                try:
                    self.model = ESMC.from_pretrained(self.model_name).to(self.device)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to load ESMC model '{self.model_name}'.\n"
                        f"Available models: esmc_300m, esmc_600m\n"
                        f"Error: {e}"
                    )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            logger.info(f"Model loaded on {self.device}")

        except ImportError as e:
            raise ImportError(
                f"Failed to import ESM models.\n"
                f"Install: pip install esm\n"
                f"Error: {e}"
            )

    @torch.no_grad()
    def extract(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract per-residue embeddings from a protein sequence.

        Args:
            sequence: Protein sequence (e.g., "MKTIIALSYIFCLVFA")

        Returns:
            Dictionary with:
                - embeddings: [L, D] per-residue embeddings
                - bos_token: [D] BOS (beginning of sequence) token embedding
                - eos_token: [D] EOS (end of sequence) token embedding
                - full_embeddings: [L+2, D] full embeddings including BOS/EOS
        """
        from esm.sdk.api import ESMProtein, LogitsConfig

        protein = ESMProtein(sequence=sequence)

        if self.model_type == "esmc":
            protein_tensor = self.model.encode(protein)
            logits_output = self.model.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
            full_embeddings = logits_output.embeddings

        elif self.model_type == "esm3":
            from esm.sdk.api import SamplingConfig

            protein_tensor = self.model.encode(protein)
            output = self.model.forward_and_sample(
                protein_tensor,
                SamplingConfig(return_per_residue_embeddings=True)
            )
            full_embeddings = output.per_residue_embedding

        # Convert to tensor
        if not isinstance(full_embeddings, torch.Tensor):
            full_embeddings = torch.tensor(full_embeddings)

        # Remove batch dimension if present
        if full_embeddings.dim() == 3:
            full_embeddings = full_embeddings.squeeze(0)

        # Move to CPU
        full_embeddings = full_embeddings.cpu()

        # Extract BOS/EOS tokens and residue embeddings
        # Full embeddings: [BOS, res1, res2, ..., resL, EOS]
        if full_embeddings.shape[0] == len(sequence) + 2:
            bos_token = full_embeddings[0]           # [D]
            eos_token = full_embeddings[-1]          # [D]
            embeddings = full_embeddings[1:-1]       # [L, D]
        else:
            # No BOS/EOS tokens (shouldn't happen, but handle it)
            bos_token = torch.zeros(full_embeddings.shape[-1])
            eos_token = torch.zeros(full_embeddings.shape[-1])
            embeddings = full_embeddings

        return {
            'embeddings': embeddings,           # [L, D]
            'bos_token': bos_token,             # [D]
            'eos_token': eos_token,             # [D]
            'full_embeddings': full_embeddings, # [L+2, D]
        }

    def extract_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from PDB file.

        Args:
            pdb_path: Path to PDB file
            chain_id: Specific chain to extract (None = all chains concatenated)

        Returns:
            Dictionary with embeddings and special tokens
        """
        sequence = self._get_sequence_from_pdb(pdb_path, chain_id)
        return self.extract(sequence)

    def extract_by_chain(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract embeddings for each chain separately.

        Args:
            pdb_path: Path to PDB file

        Returns:
            Dictionary mapping chain_id -> embeddings dict
        """
        sequences = self._get_sequences_by_chain(pdb_path)
        results = {}

        for chain_id, sequence in sequences.items():
            logger.info(f"Extracting embeddings for chain {chain_id}: {len(sequence)} residues")
            results[chain_id] = self.extract(sequence)

        return results

    def _get_sequence_from_pdb(
        self,
        pdb_path: Union[str, Path],
        chain_id: Optional[str] = None,
    ) -> str:
        """
        Extract sequence from PDB file.

        Uses PDBParser for consistent parsing across all featurizers.
        """
        from .pdb_utils import PDBParser

        parser = PDBParser(str(pdb_path))
        return parser.get_sequence(chain_id=chain_id)

    def _get_sequences_by_chain(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, str]:
        """
        Extract sequences for each chain from PDB file.

        Uses PDBParser for consistent parsing across all featurizers.
        """
        from .pdb_utils import PDBParser

        parser = PDBParser(str(pdb_path))
        return parser.get_sequence_by_chain()


class DualESMFeaturizer:
    """Extract embeddings from both ESMC and ESM3 models."""

    def __init__(
        self,
        esmc_model: str = "esmc_600m",
        esm3_model: str = "esm3-open",
        device: str = "cuda",
    ):
        """
        Initialize dual ESM extractor.

        Args:
            esmc_model: ESMC model variant
            esm3_model: ESM3 model variant
            device: "cuda" or "cpu"
        """
        logger.info("Initializing ESMC extractor...")
        self.esmc = ESMFeaturizer(
            model_type="esmc",
            model_name=esmc_model,
            device=device
        )

        logger.info("Initializing ESM3 extractor...")
        self.esm3 = ESMFeaturizer(
            model_type="esm3",
            model_name=esm3_model,
            device=device
        )

    @property
    def esmc_dim(self) -> int:
        return self.esmc.embedding_dim

    @property
    def esm3_dim(self) -> int:
        return self.esm3.embedding_dim

    def extract(self, sequence: str) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from both models.

        Returns:
            Dictionary with:
                - esmc_embeddings: [L, D1]
                - esmc_bos_token: [D1]
                - esmc_eos_token: [D1]
                - esm3_embeddings: [L, D2]
                - esm3_bos_token: [D2]
                - esm3_eos_token: [D2]
        """
        esmc_result = self.esmc.extract(sequence)
        esm3_result = self.esm3.extract(sequence)

        return {
            'esmc_embeddings': esmc_result['embeddings'],
            'esmc_bos_token': esmc_result['bos_token'],
            'esmc_eos_token': esmc_result['eos_token'],
            'esm3_embeddings': esm3_result['embeddings'],
            'esm3_bos_token': esm3_result['bos_token'],
            'esm3_eos_token': esm3_result['eos_token'],
        }

    def extract_from_pdb(
        self,
        pdb_path: Union[str, Path],
    ) -> Dict[str, torch.Tensor]:
        """Extract embeddings from PDB file using both models."""
        sequence = self.esmc._get_sequence_from_pdb(pdb_path)
        return self.extract(sequence)
