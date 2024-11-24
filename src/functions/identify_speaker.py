from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from restack_ai.function import function, log
import numpy as np
from datetime import datetime


@dataclass
class SpeakerProfile:
    id: str
    embeddings: List[torch.Tensor]
    role: str = None
    confidence_scores: List[float] = None
    last_detected: str = None
    frequency_count: int = 0


class SpeakerIdentificationSystem:
    def __init__(self):
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-sv"
        )
        self.model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-sv")
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.similarity_threshold = 0.86  # Adjustable threshold
        log.info(
            "Initialized SpeakerIdentificationSystem",
            model="microsoft/wavlm-base-sv",
            similarity_threshold=self.similarity_threshold,
        )

    def _get_embedding(self, audio_array: np.ndarray) -> torch.Tensor:
        """Generate speaker embedding from audio."""
        inputs = self.processor(audio_array, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        return F.normalize(embeddings, dim=-1).cpu()

    def add_speaker_profile(
        self, audio_array: np.ndarray, speaker_id: str, role: str = None
    ):
        """Add or update a speaker profile."""
        log.info("Adding speaker profile", speaker_id=speaker_id, role=role)
        embedding = self._get_embedding(audio_array)
        if speaker_id in self.speaker_profiles:
            log.info(
                "Updating existing speaker profile",
                speaker_id=speaker_id,
                embedding_count=len(self.speaker_profiles[speaker_id].embeddings) + 1,
            )
            self.speaker_profiles[speaker_id].embeddings.append(embedding)
            self.speaker_profiles[speaker_id].role = (
                role or self.speaker_profiles[speaker_id].role
            )
        else:
            log.info("Creating new speaker profile", speaker_id=speaker_id)
            self.speaker_profiles[speaker_id] = SpeakerProfile(
                id=speaker_id,
                embeddings=[embedding],
                role=role,
                confidence_scores=[],
                frequency_count=1,
            )

    def identify_speaker(self, audio_array: np.ndarray) -> Tuple[str, float, Dict]:
        """Identify speaker from audio and return confidence metrics."""
        log.info("Starting speaker identification")
        current_embedding = self._get_embedding(audio_array)
        best_match = None
        highest_confidence = -1
        similarity_scores = {}

        for speaker_id, profile in self.speaker_profiles.items():
            max_similarity = -1
            # Compare with all stored embeddings for this speaker
            for stored_embedding in profile.embeddings:
                similarity = torch.nn.functional.cosine_similarity(
                    current_embedding, stored_embedding
                ).item()
                max_similarity = max(max_similarity, similarity)
                log.debug(
                    "Calculated similarity score",
                    speaker_id=speaker_id,
                    max_similarity=max_similarity,
                )

            similarity_scores[speaker_id] = max_similarity
            if max_similarity > highest_confidence:
                highest_confidence = max_similarity
                best_match = speaker_id

        # Return unknown if confidence is below threshold
        if highest_confidence < self.similarity_threshold:
            log.info(
                "No speaker matched confidence threshold",
                highest_confidence=highest_confidence,
                threshold=self.similarity_threshold,
            )
            return "unknown", highest_confidence, similarity_scores

        # Update speaker profile statistics
        log.info(
            "Speaker identified",
            speaker_id=best_match,
            confidence=highest_confidence,
            frequency_count=self.speaker_profiles[best_match].frequency_count + 1,
        )
        self.speaker_profiles[best_match].frequency_count += 1
        self.speaker_profiles[best_match].confidence_scores.append(highest_confidence)

        return best_match, highest_confidence, similarity_scores


@dataclass
class FunctionInputParams:
    audio_data: Tuple[str, np.ndarray]
    command_text: str = None
    timestamp: str = None
    # timestamp: datetime = None


@function.defn()
async def process_command_audio(input: FunctionInputParams):
    print("Inside process_command_audio")
    print("Input received:", input)
    try:
        log.info("Speaker identification started", input=input)

        # Initialize speaker identification system
        speaker_system = SpeakerIdentificationSystem()

        # Process the audio
        speaker_id, confidence, similarity_scores = speaker_system.identify_speaker(
            input.audio_data[1]
        )

        # Generate command analysis
        command_analysis = {
            "timestamp": input.timestamp,
            "speaker": {
                "id": speaker_id,
                "confidence": confidence,
                "similarity_scores": similarity_scores,
                "role": speaker_system.speaker_profiles.get(speaker_id, {}).get(
                    "role", "unknown"
                ),
            },
            # "command_text": input.command_text,
            # "priority_level": _determine_priority_level(speaker_id, input.command_text),
            # "related_commands": _find_related_commands(speaker_id),
        }

        log.info("Speaker identification completed", analysis=command_analysis)
        return command_analysis

    except Exception as e:
        log.error(
            "Speaker identification failed", error=str(e), error_type=type(e).__name__
        )
        raise


def _determine_priority_level(speaker_id: str, command_text: str) -> str:
    """Determine priority level based on speaker and command content."""
    # Add logic to determine priority level
    return "HIGH" if speaker_id != "unknown" else "MEDIUM"


def _find_related_commands(speaker_id: str) -> List[Dict]:
    """Find related commands from the same speaker."""
    # Add logic to find related commands
    return []
