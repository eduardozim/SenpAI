"""
Patch defensivo para o loop Proactor do asyncio no Windows.

O Streamlit (rotas Starlette/Uvicorn) serve arquivos de mídia (st.video, st.download_button)
como um único buffer em memória. Em vídeos/pacotes grandes (> ~4 GB), o transporte Proactor
repassa o buffer inteiro para um único WSASend, que falha com:

    ValueError: buffer too large   (asyncio/windows_events.py -> ov.WSASend)

Este patch fatia escritas grandes em blocos menores, preservando a ordem dos bytes e o
controle de fluxo (pause/resume) do transporte original.
"""

import sys

_CHUNK_SIZE = 32 * 1024 * 1024  # 32 MB por chamada de WSASend


def apply_proactor_write_chunking_patch(chunk_size: int = _CHUNK_SIZE) -> bool:
    """Aplica (uma única vez por processo) o fatiamento de escritas no transporte Proactor."""
    if sys.platform != "win32":
        return False
    try:
        from asyncio import proactor_events as _pe
    except Exception:
        return False

    cls = _pe._ProactorBaseWritePipeTransport
    if getattr(cls, "_senpai_chunked_write", False):
        return True

    orig_loop_writing = cls._loop_writing

    def _chunked_loop_writing(self, f=None, data=None):
        try:
            if data is not None:
                if len(data) > chunk_size:
                    # Devolve o excedente para a FRENTE do buffer, mantendo a ordem dos bytes.
                    rest = bytearray(memoryview(data)[chunk_size:])
                    if self._buffer:
                        rest.extend(self._buffer)
                    self._buffer = rest
                    data = bytes(memoryview(data)[:chunk_size])
            elif self._buffer is not None and len(self._buffer) > chunk_size:
                data = bytes(self._buffer[:chunk_size])
                del self._buffer[:chunk_size]  # remoção no início de bytearray é O(1) amortizado
        except Exception:
            pass
        return orig_loop_writing(self, f, data)

    cls._loop_writing = _chunked_loop_writing
    cls._senpai_chunked_write = True
    return True
