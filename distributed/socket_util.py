import struct
import base64
import socket

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend


def hash_password(password: str) -> bytes:
    password_b = password.encode('utf-8')

    salt = b'geister.py'
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    key = base64.urlsafe_b64encode(kdf.derive(password_b))
    return key


class EncryptedCommunicator:
    def __init__(self, password: str) -> None:
        self.fernet = Fernet(hash_password(password))

    def send_bytes(self, sock: socket.socket, data: bytes):
        data = self.fernet.encrypt(data)
        msg = struct.pack('>I', len(data)) + data

        sock.sendall(msg)

    def recv_bytes(self, sock: socket.socket) -> bytes:
        msg_len = _recvall(sock, 4)
        msglen = struct.unpack('>I', msg_len)[0]
        msg = _recvall(sock, msglen)
        msg = self.fernet.decrypt(msg)
        return msg


def _recvall(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)
