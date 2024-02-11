from typing import TypeVar
import struct
import base64
import socket

import serde.json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

T = TypeVar('T', bound="JsonSerializable")


class JsonSerializable:
    def to_json(self) -> str:
        pass

    @classmethod
    def from_json(cls: type[T], s: str) -> T:
        pass

    def to_json_file(self, path):
        s = self.to_json()
        with open(path, mode='w') as f:
            f.write(s)

    @classmethod
    def from_json_file(cls: type[T], path) -> T:
        with open(path, mode='r') as f:
            return cls.from_json(f.read())


class SerdeJsonSerializable(JsonSerializable):
    def to_json(self) -> str:
        return serde.json.to_json(self)

    @classmethod
    def from_json(cls: type[T], s: str) -> T:
        return serde.json.from_json(cls, s)


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

    def send_str(self, sock: socket.socket, s: str, encoding: str = 'utf-8'):
        self.send_bytes(sock, s.encode(encoding))

    def recv_str(self, sock: socket.socket, encoding: str = 'utf-8'):
        return self.recv_bytes(sock).decode(encoding)

    def send_json_obj(self, sock: socket.socket, obj: JsonSerializable):
        json_str = obj.to_json()
        self.send_str(sock, json_str)

    def recv_json_obj(self, sock: socket.socket, ty: type[T]) -> T:
        json_str = self.recv_str(sock)
        return ty.from_json(json_str)


def _recvall(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)
