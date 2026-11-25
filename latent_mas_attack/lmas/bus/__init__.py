"""OS-level tensor IPC bus and edge reference monitor for latent MAS handoffs."""

from .shm_bus import SharedMemoryTensorBus, SerializedBytesBus, BusTamperError, HandoffHeader
from .edge_monitor import EdgeTensorReferenceMonitor, EdgeInspectionResult

__all__ = [
    "SharedMemoryTensorBus",
    "SerializedBytesBus",
    "BusTamperError",
    "HandoffHeader",
    "EdgeTensorReferenceMonitor",
    "EdgeInspectionResult",
]
