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

const vector<vector<ulong>> DISTANCE_MASKS_P = {
	{
		0b100000000000000000000000000000000000ULL,
        0b010000100000000000000000000000000000ULL,
        0b001000010000100000000000000000000000ULL,
        0b000100001000010000100000000000000000ULL,
        0b000010000100001000010000100000000000ULL,
        0b000001000010000100001000010000100000ULL,
	},
	{
		0b000001000000000000000000000000000000ULL,
        0b000010000001000000000000000000000000ULL,
        0b000100000010000001000000000000000000ULL,
        0b001000000100000010000001000000000000ULL,
        0b010000001000000100000010000001000000ULL,
        0b100000010000001000000100000010000001ULL,
	}
};

const vector<vector<ulong>> DISTANCE_MASKS_O = {
	{
		0b100000ULL,
        0b100000010000ULL,
        0b100000010000001000ULL,
        0b100000010000001000000100ULL,
        0b100000010000001000000100000010ULL,
        0b100000010000001000000100000010000001ULL,
	},
	{
		0b000001ULL,
        0b000001000010ULL,
        0b000001000010000100ULL,
        0b000001000010000100001000ULL,
        0b000001000010000100001000010000ULL,
        0b000001000010000100001000010000100000ULL,
	}
};

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

struct MoveResult {
	int id, color;

	MoveResult(int id, int color) {
		this->id = id;
		this->color = color;
	}
};

struct SearchParam {
	vector<int> pos_o, color_o;
	vector<vector<ulong>> escape_distance_mask_p;
	vector<vector<ulong>> escape_distance_mask_o;

	SearchParam() {

	}

	SearchParam(vector<int> pos_o, vector<int> color_o, int root_player) {
		this->pos_o = pos_o;
		this->color_o = color_o;

		if(root_player == 1) {
			this->escape_distance_mask_p = DISTANCE_MASKS_P;
			this->escape_distance_mask_o = DISTANCE_MASKS_O;
		}
		else {
			this->escape_distance_mask_p = DISTANCE_MASKS_O;
			this->escape_distance_mask_o = DISTANCE_MASKS_P;
		}
	}

	MoveResult apply_move(ulong move, int d, int player) {
		int pos = tzcnt(move);

		if(player == 1) {
			int id = index_of(&this->pos_o, pos + DIRECTIONS[d]);

			if(id != -1) {
				int color = this->color_o[id];
				this->pos_o.at(id) = -1;
				this->color_o.at(id) = 0;
				return {id, color};
			}
		}
		else {
			int id = index_of(&this->pos_o, pos);
			this->pos_o.at(id) = pos + DIRECTIONS[d];
		}

		return {-1, -1};
	}

	void undo_apply_move(ulong move, int d, int player, MoveResult result) {
		int pos = tzcnt(move);

		if(player == 1) {
			if(result.id != -1) {
				this->pos_o.at(result.id) = pos + DIRECTIONS[d];
				this->color_o.at(result.id) = result.color;
			}
		}
		else {
			int id = index_of(&this->pos_o, pos + DIRECTIONS[d]);
			this->pos_o.at(id) = pos;
		}
	}
};

struct SolveResult
{
	int eval;
	int cause_piece_id;
	int cause_piece_color;

	SolveResult(int eval, int cause_piece_id, int cause_piece_color) {
		this->eval = eval;
		this->cause_piece_id = cause_piece_id;
		this->cause_piece_color = cause_piece_color;
	}
};

SolveResult get_max_result(SolveResult r1, SolveResult r2, int player) {
	if(r1.eval == r2.eval) {
		if(r2.cause_piece_color == 1)
			return r2;
		return r1;
	}
	return r1.eval * player > r2.eval * player ? r1 : r2;
}

SolveResult SOLVE_RESULT_NONE = {0, -1, -1};

int calc_min_distance(vector<ulong> distance_mask, ulong pieces) {
	for(int i = 0; i < 6; i++) {
		if (distance_mask[i] & pieces)
			return i;
	}
	return 6;
}

bool is_escaped_p(SearchParam* search, Board* b) {
	for (int i = 0; i < 2; i++) {
		int distance_pb = calc_min_distance(search->escape_distance_mask_p[i], b->pb);
		int distance_pr = calc_min_distance(search->escape_distance_mask_p[i], b->pr);
		int distance_o = calc_min_distance(search->escape_distance_mask_p[i], b->o);

		if(distance_pb < distance_o && distance_pb <= distance_pr) {
			return true;
		}
	}
	return false;
}

bool is_escaped_o(SearchParam* search, Board* b, int* escaped_id) {
	ulong o_b = 0;
	ulong o_r = 0;

	for (int i = 0; i < 8; i++) {
		if (search->color_o[i] == 0)
			o_r |= 1ULL << search->pos_o[i];
		else
			o_b |= 1ULL << search->pos_o[i];
	}

	for(int i = 0; i < 2; i++) {
		int distance_p = calc_min_distance(search->escape_distance_mask_o[i], b->pb | b->pr);
		int distance_ob = calc_min_distance(search->escape_distance_mask_o[i], o_b);
		int distance_or = calc_min_distance(search->escape_distance_mask_o[i], o_r);

		if(distance_ob < distance_p && distance_ob <= distance_or) {
			ulong escaped_mask = o_b & search->escape_distance_mask_o[i][distance_ob];
			*escaped_id = index_of(&search->pos_o, tzcnt(escaped_mask));

			return true;
		}
	}
	return false;
}

bool is_done(SearchParam* search, Board* b, int player, int* winner, int* type, int* escaped_id) {
	if (player == 1) {
		if (is_escaped_p(search, b)) {
			*winner = 1;
			*type = WIN_ESCAPE;
			return true;
		}

		/*if ((b->pb & search->escape_mask_p) != 0) {
			*winner = 1;
			*type = WIN_ESCAPE;
			return true;
		}*/
	}
	else {
		if (is_escaped_o(search, b, escaped_id)) {
			*winner = -1;
			*type = WIN_ESCAPE;
			return true;
		}

		/*ulong escaped = b->o & search->escape_mask_o;

		if (escaped != 0) {
			int id = index_of(&search->pos_o, tzcnt(escaped));

			if(search->color_o[id] != 0) {
				*winner = -1;
				*type = WIN_ESCAPE;
				*escaped_id = index_of(&search->pos_o, tzcnt(escaped));
				return true;
			}
		}*/
	}

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

	int n_cap_red = 0;
	for(int i = 0; i < 8; i++) {
		if(search->pos_o[i] == -1 && search->color_o[i] == 0)
			n_cap_red++;
	}

	if (n_cap_red >= 4) {
		*winner = -1;
		*type = WIN_RED4;
		return true;
	}

	*winner = 0;
	*type = WIN_NONE;
	return false;
}

SolveResult solve(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	int winner = 0;
	int type = 0;
	int escaped_id = -1;

	if (is_done(search, board, player, &winner, &type, &escaped_id)) {
		if(type == WIN_ESCAPE)
			return {winner * (depth + 1), escaped_id, 1};

		else if(type == WIN_RED4 && winner == -1)
			return {winner * (depth + 1), -1, 0};

		else
			return SOLVE_RESULT_NONE;
	}

	if (depth <= 0)
		return SOLVE_RESULT_NONE;

	vector<ulong> moves = get_moves(board, player);

	vector<SolveResult> results;

	Board* next_board = new Board(0, 0, 0);
	SolveResult max_result = {-1000000 * player, -1, -1};
	ulong move;

	for (int d = 0; d < 4; d++) {
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0) {
			moves_d = moves_d ^ move;

			MoveResult m_result = search->apply_move(move, d, player);

			step(board, next_board, move, d);

			SolveResult result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

			search->undo_apply_move(move, d, player, m_result);

			if (result.eval == -depth && result.cause_piece_id == -1 && result.cause_piece_color == 0) {
				result.cause_piece_id = m_result.id;
			}

			results.push_back(result);

			max_result = get_max_result(max_result, result, player);
			alpha = max(alpha, result.eval * player);

			if (alpha >= beta) {
				delete next_board;
				return max_result;
			}
		}
	}

	delete next_board;

	if (max_result.eval * player >= 0 || max_result.eval > 0)
		return max_result;
	
	int colors[] = {-1, -1, -1, -1, -1, -1, -1, -1};

	for(SolveResult result: results) {
		int id = result.cause_piece_id;
		int color = result.cause_piece_color;

		if (colors[id] != -1 && colors[id] != color)
			return SOLVE_RESULT_NONE;

		colors[id] = color;
	}
	return max_result;
}

int solve_root(SearchParam* search, Board* board, int alpha, int beta, int player, int depth, ulong* max_move, int* max_d, int* escaped_id) {
	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0, 0);
	int max_e = -1000000;
	ulong move;

	for (int d = 0; d < 4; d++) {
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0) {
			moves_d = moves_d ^ move;

			MoveResult m_result = search->apply_move(move, d, player);

			step(board, next_board, move, d);
			SolveResult result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

			search->undo_apply_move(move, d, player, m_result);

			if (result.eval * player > max_e) {
				max_e = result.eval * player;
				*max_d = d;
				*max_move = move;
				*escaped_id = result.cause_piece_id;
			}
			alpha = max(alpha, result.eval * player);
		}
	}
	delete next_board;

	return max_e * player;
}


py::tuple find_checkmate(py::array_t<int> pos_p, py::array_t<int> color_p,
						 py::array_t<int> pos_o, py::array_t<int> color_o,
						 int turn_player, int player, int depth) {
	ulong pb = 0;
	ulong pr = 0;
	ulong o = 0;

	vector<int> pos_o_v, color_o_v;

	for (int i = 0; i < 8; i++) {
		int p_i = *pos_p.data(i);
		int c_p_i = *color_p.data(i);

		if (p_i >= 0) {
			if (c_p_i == 1)
				pb |= x_to_bit(p_i);
			else
				pr |= x_to_bit(p_i);
		}

		int o_i = *pos_o.data(i);
		int c_o_i = *color_o.data(i);
		pos_o_v.push_back(o_i);
		color_o_v.push_back(c_o_i);

		if (o_i >= 0)
			o |= x_to_bit(o_i);
	}

	SearchParam search = {pos_o_v, color_o_v, player};

	Board* board = new Board(pb, pr, o);

	ulong max_move;
	int max_d;
	int escaped_id;

	int e = solve_root(&search, board, -100, 100, turn_player, depth, &max_move, &max_d, &escaped_id);

	delete board;

	if (e == 0) {
		return py::make_tuple(-1, e, -1);
	}

	int pos = tzcnt(max_move);

	for (int i = 0; i < 8; i++) {
		int p_i;
		if(turn_player == 1)
			p_i = *pos_p.data(i);
		else
			p_i = *pos_o.data(i);

		if (p_i == pos) {
			int action = i * 4 + max_d;
			return py::make_tuple(action, e, escaped_id);
		}
	}
	return py::make_tuple(-1, e, -1);
}

PYBIND11_MODULE(checkmate_lib_old, m) {
	m.def("find_checkmate", &find_checkmate);
}
