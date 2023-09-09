#include <iostream>
#include <vector>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define ulong unsigned long long

#define popcount __builtin_popcountll
#define x_to_bit(x) (1ULL << (x))

ulong first_bit(ulong b) {
	return b & (~b + 1);
}

using namespace std;

struct Board {
	ulong pb, pr, o;

	Board(ulong pb, ulong pr, ulong o) {
		this->pb = pb;
		this->pr = pr;
		this->o = o;
	}
};

string b_to_string(Board* b) {
	const string line = "+---+---+---+---+---+---+";
	string s = line + "\r\n";

	for (int i = 0; i < 36; i++) {
		if (i % 6 == 0 && i != 0) {
			s += "|\r\n" + line + "\r\n";
		}

		if (b->pb & x_to_bit(i)) {
			s += "| B ";
		}
		else if (b->pr & x_to_bit(i)) {
			s += "| R ";
		}
		else if (b->o & x_to_bit(i)) {
			s += "| O ";
		}
		else {
			s += "|   ";
		}
	}
	s += "|\r\n" + line + "\r\n";

	return s;
}

const ulong BOARD_MASK = (1ULL << 36) - 1;
const ulong MASKS[] = {
	BOARD_MASK,
	BOARD_MASK & 0x7df7df7df,
	BOARD_MASK & 0xfbefbefbe,
	BOARD_MASK
};

const int DIRECTIONS[] = {-6, -1, 1, 6};

const ulong ESCAPE_MASK_P = 0b100001000000000000000000000000000000ULL;
const ulong ESCAPE_MASK_O = 0b100001ULL;

const ulong INIT_MASK_P = 0b011110011110ULL;
const ulong INIT_MASK_O = 0b011110011110000000000000000000000000ULL;

static ulong GetMoveShiftL(ulong p, ulong mask, int shift) {
	return ((p << shift) & mask) >> shift;
}

static ulong GetMoveShiftR(ulong p, ulong mask, int shift) {
	return ((p >> shift) & mask) << shift;
}

static vector<ulong> GetMoves(ulong p) {
	ulong move0 = GetMoveShiftR(p, MASKS[0] & ~p, 6);
	ulong move1 = GetMoveShiftR(p, MASKS[1] & ~p, 1);
	ulong move2 = GetMoveShiftL(p, MASKS[2] & ~p, 1);
	ulong move3 = GetMoveShiftL(p, MASKS[3] & ~p, 6);

	return vector<ulong>{ move0, move1, move2, move3 };
}

static vector<ulong> get_moves(Board* b, int player) {
	if (player == 1)
		return GetMoves(b->pb | b->pr);
	else
		return GetMoves(b->o);
}

static ulong shift_left(ulong b, int shift) {
	if (shift > 0)
		return b << shift;
	else
		return b >> -shift;
}

void step(Board* b, Board* next_board, ulong move, int d) {
	ulong next = shift_left(move, DIRECTIONS[d]);
	ulong diff = move | next;

	// cout << b_to_string(new Board(move, next, 0));

	if ((b->pb & move) != 0) {
		next_board->pb = b->pb ^ diff;
		next_board->pr = b->pr;
		next_board->o = b->o & ~next;
		return;
	}
	else if ((b->pr & move) != 0) {
		next_board->pb = b->pb;
		next_board->pr = b->pr ^ diff;
		next_board->o = b->o & ~next;
		return;
	}
	else if ((b->o & move) != 0) {
		next_board->pb = b->pb & ~next;
		next_board->pr = b->pr & ~next;
		next_board->o = b->o ^ diff;
		return;
	}
}

const int WIN_NONE = 0;
const int WIN_ESCAPE = 1;
const int WIN_BLUE4 = 2;
const int WIN_RED4 = 3;

struct SearchParam {
	int n_cap_ob;
	ulong escape_mask_p, escape_mask_o;

	SearchParam(int n_cap_ob, ulong escape_mask_p, ulong escape_mask_o) {
		this->n_cap_ob = n_cap_ob;
		
		this->escape_mask_p = escape_mask_p;
		this->escape_mask_o = escape_mask_o;
	}
};

bool is_done(SearchParam* search, Board* b, int player, int* winner, int* type) {
	if (b->pb == 0) {
		*winner = -1;
		*type = WIN_BLUE4;
		return true;
	}

	if (b->pr == 0) {
		*winner = 1;
		*type = WIN_RED4;
		return true;
	}

	if (8 - popcount(b->o) - search->n_cap_ob >= 4) {
		*winner = -1;
		*type = WIN_RED4;
		return true;
	}

	if (player == 1) {
		if ((b->pb & search->escape_mask_p) != 0) {
			*winner = 1;
			*type = WIN_ESCAPE;
			return true;
		}
	}
	else {
		if ((b->o & search->escape_mask_o) != 0) {
			*winner = -1;
			*type = WIN_ESCAPE;
			return true;
		}
	}
	*winner = 0;
	*type = WIN_NONE;
	return false;
}

int solve(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	int winner = 0;
	int type = 0;

	if (is_done(search, board, player, &winner, &type))
		return winner * player * (depth + 1);

	if (depth <= 0)
		return 0;

	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0, 0);
	int max_e = -1000000;
	ulong move;

	for (int d = 0; d < 4; d++)
	{
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0)
		{
			moves_d = moves_d ^ move;

			//cout << static_cast<std::bitset<36>>(move) << ", " << d << endl;
			step(board, next_board, move, d);
			//cout << b_to_string(board);
			//cout << b_to_string(next_board);
			int e = -solve(search, next_board, -beta, -alpha, -player, depth - 1);

			max_e = max(max_e, e);
			alpha = max(alpha, e);

			if (alpha >= beta)
			{
				delete next_board;
				return max_e;
			}
		}
	}
	delete next_board;

	return max_e;
}

int solve_root(SearchParam* search, Board* board, int alpha, int beta, int player, int depth, ulong* max_move, int* max_d) {
	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0, 0);
	int max_e = -1000000;
	ulong move;

	for (int d = 0; d < 4; d++)
	{
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0)
		{
			moves_d = moves_d ^ move;

			step(board, next_board, move, d);
			int e = -solve(search, next_board, -beta, -alpha, -player, depth - 1);

			if (e > max_e) {
				max_e = e;
				*max_d = d;
				*max_move = move;
			}
			alpha = max(alpha, e);
		}
	}
	delete next_board;

	return max_e;
}

ulong tzcnt(ulong n)
{
	return(popcount(~n & (n - 1)));
}

int find_checkmate(py::array_t<int> pieces_p, py::array_t<int> colors_p, py::array_t<int> pieces_o, int n_cap_ob, int player, int depth) {
	ulong pb = 0;
	ulong pr = 0;
	ulong o = 0;

	for (int i = 0; i < 8; i++) {
		int p_i = *pieces_p.data(i);
		int c_i = *colors_p.data(i);

		if (p_i >= 0) {
			if (c_i == 1)
				pb |= x_to_bit(p_i);
			else
				pr |= x_to_bit(p_i);
		}

		int o_i = *pieces_o.data(i);
		if (o_i >= 0)
			o |= x_to_bit(o_i);
	}

	SearchParam* search;

	if (player == 1) {
		search = new SearchParam(n_cap_ob, ESCAPE_MASK_P, ESCAPE_MASK_O);
	}
	else {
		search = new SearchParam(n_cap_ob, ESCAPE_MASK_O, ESCAPE_MASK_P);
	}

	Board* board = new Board(pb, pr, o);

	ulong max_move;
	int max_d;

	int e = solve_root(search, board, -100, 100, 1, depth, &max_move, &max_d);

	delete search;
	delete board;

	if (e <= 0)
		return -1;

	int pos = tzcnt(max_move);

	for (int i = 0; i < 8; i++) {
		int p_i = *pieces_p.data(i);

		if (p_i == pos)
			return i * 4 + max_d;
	}

	return -1;
}

PYBIND11_MODULE(geister_lib, m) {
	m.doc() = "pybind11 example";
	m.def("find_checkmate", &find_checkmate);
}

/*int main()
{
	ulong pb = 0b000000'100000'000000'010000'000000'000000ULL;
	ulong pr = 0b000000'010000'000000'100000'000000'000000ULL;
	ulong o = 0b010001'000011'000001'000000'000000'000000ULL;
	Board* board = new Board(pb, pr, o);
	cout << b_to_string(board) << endl;

	SearchParam* search = new SearchParam(3, 0, 0);

	auto start = std::chrono::system_clock::now();

	int e = solve(search, board, 0, 100, 1, 7);

	auto end = std::chrono::system_clock::now();
	double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

	cout << "Eval : " << e << endl;
	cout << "Time : " << time << "\r\n";

	return 0;
}*/
