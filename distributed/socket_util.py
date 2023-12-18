import socket
import struct
import asyncio


def send_msg(sock: socket.socket, msg: bytes):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    sock.sendall(msg)


def recv_msg(sock: socket.socket) -> bytes:
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(sock, msglen)


def recvall(sock: socket.socket, n: int) -> bytes:
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


async def send_msg_async(loop: asyncio.AbstractEventLoop, sock: socket.socket, msg: bytes):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    await loop.sock_sendall(sock, msg)


async def recv_msg_async(loop: asyncio.AbstractEventLoop, sock: socket.socket) -> bytes:
    # Read message length and unpack it into an integer
    raw_msglen = await recvall_async(loop, sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return await recvall_async(loop, sock, msglen)


async def recvall_async(loop: asyncio.AbstractEventLoop, sock: socket.socket, n: int) -> bytes:
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = await loop.sock_recv(sock, n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
