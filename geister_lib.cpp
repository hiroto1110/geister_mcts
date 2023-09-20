#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#define ulong unsigned long long

#define popcount __builtin_popcountll
// #define popcount __popcnt64
#define x_to_bit(x) (1ULL << (x))

ulong first_bit(ulong b) {
	return b & (~b + 1);
}

ulong tzcnt(ulong n)
{
	return(popcount(~n & (n - 1)));
}

using namespace std;

int index_of(vector<int>* v, int element) {
	for (int i = 0; i < 8; i++) {
		if (v->at(i) == element)
			return i;
	}
	return -1;
}

string vector_to_string(vector<int>* vec) {
	std::stringstream ss;
	for (auto it = vec->begin(); it != vec->end(); it++) {
		if (it != vec->begin()) {
			ss << " ";
		}
		ss << *it;
	}
	return ss.str();
}

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
	vector<int> pos_p, pos_o;
	int n_cap_ob;
	ulong escape_mask_p, escape_mask_o;

	SearchParam(vector<int> pos_p, vector<int> pos_o, int n_cap_ob, ulong escape_mask_p, ulong escape_mask_o) {
		this->pos_p = pos_p;
		this->pos_o = pos_o;

		this->n_cap_ob = n_cap_ob;
		
		this->escape_mask_p = escape_mask_p;
		this->escape_mask_o = escape_mask_o;
	}

	void apply_move(ulong move, int d, int player) {
		vector<int>* pos_v = player == 1 ? &this->pos_p : &this->pos_o;
		int pos = tzcnt(move);

		//cout << vector_to_string(pos_v) << ", " << pos << ", " << d << ": do" << endl;
		int id = index_of(pos_v, pos);

		pos_v->at(id) = pos + DIRECTIONS[d];
	}

	void undo_apply_move(ulong move, int d, int player) {
		vector<int>* pos_v = player == 1 ? &this->pos_p : &this->pos_o;
		int pos = tzcnt(move);

		//cout << vector_to_string(pos_v) << ", " << pos << ", " << d << ": undo" << endl;
		int id = index_of(pos_v, pos + DIRECTIONS[d]);
		if (id == -1) {
			//cout << vector_to_string(pos_v) << ", " << pos << ", " << d << ": undo" << endl;
		}

		pos_v->at(id) = pos;
	}
};

struct SolveResult
{
	int eval;
	int escaped_id;

	SolveResult(int eval, int escaped_id) {
		this->eval = eval;
		this->escaped_id = escaped_id;
	}
};

SolveResult get_max_result(SolveResult r1, SolveResult r2, int player) {
	return r1.eval * player > r2.eval * player ? r1 : r2;
}

SolveResult SOLVE_RESULT_NONE = {0, -1};

bool is_done(SearchParam* search, Board* b, int player, int* winner, int* type, int* escaped_id) {
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
		ulong escaped = b->o & search->escape_mask_o;

		if (escaped != 0) {
			*winner = -1;
			*type = WIN_ESCAPE;
			*escaped_id = index_of(&search->pos_o, tzcnt(escaped));
			return true;
		}
	}
	*winner = 0;
	*type = WIN_NONE;
	return false;
}

SolveResult solve(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	int winner = 0;
	int type = 0;
	int escaped_id = -1;

	if (is_done(search, board, player, &winner, &type, &escaped_id))
		return { winner * (depth + 1),  escaped_id };

	if (depth <= 0)
		return SOLVE_RESULT_NONE;

	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0, 0);
	SolveResult max_result = {-1000000 * player, -1};
	ulong move;

	for (int d = 0; d < 4; d++)
	{
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0)
		{
			moves_d = moves_d ^ move;

			search->apply_move(move, d, player);

			step(board, next_board, move, d);
			SolveResult result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

			search->undo_apply_move(move, d, player);

			max_result = get_max_result(max_result, result, player);
			alpha = max(alpha, result.eval * player);

			if (alpha >= beta)
			{
				delete next_board;
				return max_result;
			}
		}
	}
	delete next_board;
	return max_result;
}

int solve_root(SearchParam* search, Board* board, int alpha, int beta, int player, int depth, ulong* max_move, int* max_d, int* escaped_id) {
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

			search->apply_move(move, d, player);

			step(board, next_board, move, d);
			SolveResult result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

			search->undo_apply_move(move, d, player);

			if (result.eval * player > max_e) {
				max_e = result.eval * player;
				*max_d = d;
				*max_move = move;
				*escaped_id = result.escaped_id;
			}
			alpha = max(alpha, result.eval * player);
		}
	}
	delete next_board;

	return max_e * player;
}

py::tuple find_checkmate(py::array_t<int> pieces_p, py::array_t<int> colors_p, py::array_t<int> pieces_o, int n_cap_ob, int turn_player, int player, int depth) {
	ulong pb = 0;
	ulong pr = 0;
	ulong o = 0;

	vector<int> pos_p;
	vector<int> pos_o;

	for (int i = 0; i < 8; i++) {
		int p_i = *pieces_p.data(i);
		int c_i = *colors_p.data(i);

		pos_p.push_back(p_i);

		if (p_i >= 0) {
			if (c_i == 1)
				pb |= x_to_bit(p_i);
			else
				pr |= x_to_bit(p_i);
		}

		int o_i = *pieces_o.data(i);
		pos_o.push_back(o_i);

		if (o_i >= 0)
			o |= x_to_bit(o_i);
	}

	SearchParam* search;

	if (player == 1) {
		search = new SearchParam(pos_p, pos_o, n_cap_ob, ESCAPE_MASK_P, ESCAPE_MASK_O);
	}
	else {
		search = new SearchParam(pos_p, pos_o, n_cap_ob, ESCAPE_MASK_O, ESCAPE_MASK_P);
	}

	Board* board = new Board(pb, pr, o);

	ulong max_move;
	int max_d;
	int escaped_id;

	int e = solve_root(search, board, -100, 100, turn_player, depth, &max_move, &max_d, &escaped_id);

	delete search;
	delete board;

	if (e == 0)
		return py::make_tuple(-1, e, -1);

	int pos = tzcnt(max_move);

	for (int i = 0; i < 8; i++) {
		int p_i = *pieces_p.data(i);

		if (p_i == pos)
		{
			int action = i * 4 + max_d;
			return py::make_tuple(action, e, escaped_id);
		}
	}
	return py::make_tuple(-1, e, -1);
}

PYBIND11_MODULE(geister_lib, m) {
	m.def("find_checkmate", &find_checkmate);
}

/*
int main()
{
	ulong pb = 0b000000'100000'000000'010000'000000'000000ULL;
	ulong pr = 0b000000'010000'000000'100000'000000'000000ULL;
	ulong ob = 0b010001'000011'000000'000000'000000'000000ULL;
	ulong o_r = 0b000000'000000'000001'000000'000000'000000ULL;
	// Board* board = new Board(pb, pr, ob | o_r);
	Board* board = new Board(ob, o_r, pb | pr);
	cout << b_to_string(board) << endl;

	vector<int> pos_o = vector<int>{16, 17, 28, 29};
	vector<int> pos_p = vector<int>{18, 24, 25, 30, 34};
	SearchParam* search = new SearchParam(pos_p, pos_o, 3, ESCAPE_MASK_O, ESCAPE_MASK_P);

	auto start = std::chrono::system_clock::now();

	SolveResult result = solve(search, board, -100, 100, 1, 7);

	auto end = std::chrono::system_clock::now();
	double time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0);

	cout << "Eval : " << result.eval << endl;
	cout << "Escaped ID : " << result.escaped_id << endl;
	cout << "Time : " << time << "\r\n";

	return 0;
}
*/