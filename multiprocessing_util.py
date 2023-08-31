import multiprocessing as mp


class MultiSenderPipe:
    def __init__(self, n_senders) -> None:
        self.n_senders = n_senders
        self.pipes = [mp.Pipe(duplex=False) for _ in range(n_senders)]

    def poll(self):
        return any([reciver.poll() for reciver, _ in self.pipes])

    def recv(self):
        for reciver, _ in self.pipes:
            if not reciver.poll():
                continue

            return reciver.recv()

    def get_sender(self, i):
        return self.pipes[i][1]
