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
	ulong p_b, p_r, o_b, o_r, o_u;

	Board(ulong p_b, ulong p_r, ulong o_b, ulong o_r, ulong o_u) {
		this->p_b = p_b;
		this->p_r = p_r;
		this->o_b = o_b;
		this->o_r = o_r;
		this->o_u = o_u;
	}
};

string b_to_string(Board* b) {
	const string line = "+---+---+---+---+---+---+";
	string s = line + "\r\n";
	ulong o = b->o_b | b->o_r | b->o_u;

	for (int i = 0; i < 36; i++) {
		if (i % 6 == 0 && i != 0) {
			s += "|\r\n" + line + "\r\n";
		}

		if (b->p_b & x_to_bit(i)) {
			s += "| B ";
		}
		else if (b->p_r & x_to_bit(i)) {
			s += "| R ";
		}
		else if (o & x_to_bit(i)) {
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
		return GetMoves(b->p_b | b->p_r);
	else
		return GetMoves(b->o_b | b->o_r | b->o_u);
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

	if ((b->p_b & move) != 0) {
		next_board->p_b = b->p_b ^ diff;
		next_board->p_r = b->p_r;
		next_board->o_b = b->o_b & ~next;
		next_board->o_r = b->o_r & ~next;
		next_board->o_u = b->o_u & ~next;
		return;
	}
	else if ((b->p_r & move) != 0) {
		next_board->p_b = b->p_b;
		next_board->p_r = b->p_r ^ diff;
		next_board->o_b = b->o_b & ~next;
		next_board->o_r = b->o_r & ~next;
		next_board->o_u = b->o_u & ~next;
		return;
	}
	else if ((b->o_b & move) != 0) {
		next_board->p_b = b->p_b & ~next;
		next_board->p_r = b->p_r & ~next;
		next_board->o_b = b->o_b ^ diff;
		next_board->o_r = b->o_r;
		next_board->o_u = b->o_u;
		return;
	}
	else if ((b->o_r & move) != 0) {
		next_board->p_b = b->p_b & ~next;
		next_board->p_r = b->p_r & ~next;
		next_board->o_b = b->o_b;
		next_board->o_r = b->o_r ^ diff;
		next_board->o_u = b->o_u;
		return;
	}
	else if ((b->o_u & move) != 0) {
		next_board->p_b = b->p_b & ~next;
		next_board->p_r = b->p_r & ~next;
		next_board->o_b = b->o_b;
		next_board->o_r = b->o_r;
		next_board->o_u = b->o_u ^ diff;
		return;
	}
}

const int WIN_NONE = 0;
const int WIN_ESCAPE = 1;
const int WIN_BLUE4 = 2;
const int WIN_RED4 = 3;
const int END_CAP7 = 4;

struct SearchParam {
	int root_player;
	int init_cap_o_b_cnt, init_cap_o_r_cnt;
	int init_o_b_cnt, init_o_r_cnt, init_o_u_cnt;
	vector<vector<ulong>> escape_distance_mask_p;
	vector<vector<ulong>> escape_distance_mask_o;

	SearchParam() {

	}

	SearchParam(int root_player, ulong o_b, ulong o_r, ulong o_u, int init_cap_o_b_cnt, int init_cap_o_r_cnt) {
		this->root_player = root_player;
		if(root_player == 1) {
			this->escape_distance_mask_p = DISTANCE_MASKS_P;
			this->escape_distance_mask_o = DISTANCE_MASKS_O;
		}
		else {
			this->escape_distance_mask_p = DISTANCE_MASKS_O;
			this->escape_distance_mask_o = DISTANCE_MASKS_P;
		}

		this->init_o_b_cnt = popcount(o_b);
		this->init_o_r_cnt = popcount(o_r);
		this->init_o_u_cnt = popcount(o_u);
		this->init_cap_o_b_cnt = init_cap_o_b_cnt;
		this->init_cap_o_r_cnt = init_cap_o_r_cnt;
	}

	int n_cap_blue_in_best_case(Board* b) {
		return this->init_cap_o_b_cnt + this->init_o_b_cnt + this->init_o_u_cnt - popcount(b->o_b | b->o_u);
	}

	int n_cap_red_in_worst_case(Board* b) {
		return this->init_cap_o_r_cnt + this->init_o_r_cnt + this->init_o_u_cnt - popcount(b->o_r | b->o_u);
	}
};

bool test_f = false;

struct SolveResult
{
	int eval;
	ulong cause_piece_mask;
	int cause_piece_color;

	SolveResult(int eval, ulong cause_piece_mask, int cause_piece_color) {
		this->eval = eval;
		this->cause_piece_mask = cause_piece_mask;
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

SolveResult SOLVE_RESULT_NONE = {0, 0, -1};

int calc_min_distance(vector<ulong>* distance_mask, ulong pieces) {
	if(pieces == 0)
		return 6;

	for(int i = 0; i < 6; i++) {
		if (distance_mask->at(i) & pieces)
			return i;
	}
	return 6;
}

bool is_escaped_root_p(SearchParam* search, Board* b, int player, int i, int* action, int* distance, ulong* escaped_mask) {
	vector<ulong>* masks = &search->escape_distance_mask_p[i];

	int distance_p_b = calc_min_distance(masks, b->p_b);
	int distance_p_r = calc_min_distance(masks, b->p_r);
	int distance_o = calc_min_distance(masks, b->o_b | b->o_r | b->o_u);

	int offset = player == 1 ? 0 : -1;

	if(distance_p_b < distance_o + offset && distance_p_b <= distance_p_r) {
		*distance = distance_p_b;

		ulong mask = b->p_b & masks->at(distance_p_b);
		*escaped_mask = first_bit(mask);

		if (action != nullptr) {
			int pos = tzcnt(*escaped_mask);
			*action = pos * 4;

			if(pos % 6 == 0 || pos % 6 == 5) {
				if(search->root_player == 1)
					*action += 3;
				else
					*action += 0;
			}
			else {
				if(i == 0)
					*action += 2;
				else
					*action += 1;
			}
		}
		return true;
	}
	return false;
}

bool is_escaped_root_p(SearchParam* search, Board* b, int player, int* action, int* distance, ulong* escaped_mask) {
	int d1 = 6, d2 = 6;
	int a1 = 0, a2 = 0;
	ulong m1 = 0, m2 = 0;
	bool esc1 = is_escaped_root_p(search, b, player, 0, &a1, &d1, &m1);
	bool esc2 = is_escaped_root_p(search, b, player, 1, &a2, &d2, &m2);

	if (!esc1 && !esc2)
		return false;

	if(d1 < d2) {
		*distance = d1;
		*action = a1;
		*escaped_mask = m1;
	}
	else {
		*distance = d2;
		*action = a2;
		*escaped_mask = m2;
	}
	return true;
}

bool is_escaped_root_o(SearchParam* search, Board* b, int player, int i, int* action, int* distance, ulong* escaped_mask) {
	vector<ulong>* masks = &search->escape_distance_mask_o[i];

	int distance_p = calc_min_distance(masks, b->p_b | b->p_r);
	int distance_ob = calc_min_distance(masks, b->o_b | b->o_u);
	int distance_or = calc_min_distance(masks, b->o_r);

	int offset = player == -1 ? 0 : -1;

	if(distance_ob < distance_p + offset && distance_ob <= distance_or) {
		*distance = distance_ob;

		ulong mask = (b->o_b | b->o_u) & masks->at(distance_ob);
		*escaped_mask = first_bit(mask);

		if (action != nullptr) {
			int pos = tzcnt(*escaped_mask);
			*action = pos * 4;

			if(pos % 6 == 0 || pos % 6 == 5) {
				if(search->root_player == 1)
					*action += 0;
				else
					*action += 3;
			}
			else {
				if(i == 0)
					*action += 2;
				else
					*action += 1;
			}
		}
		return true;
	}
	return false;
}

bool is_escaped_root_o(SearchParam* search, Board* b, int player, int* action, int* distance, ulong* escaped_mask) {
	int d1 = 6, d2 = 6;
	int a1 = 0, a2 = 0;
	ulong m1 = 0, m2 = 0;
	bool esc1 = is_escaped_root_o(search, b, player, 0, &a1, &d1, &m1);
	bool esc2 = is_escaped_root_o(search, b, player, 1, &a2, &d2, &m2);

	if (!esc1 && !esc2)
		return false;

	if(d1 < d2) {
		*distance = d1;
		*action = a1;
		*escaped_mask = m1;
	}
	else {
		*distance = d2;
		*action = a2;
		*escaped_mask = m2;
	}
	return true;
}

bool is_escaped_root(SearchParam* search, Board* b, int player, int* winner, int* action, int* escaped_depth, ulong* escaped_mask) {
	int d1 = 6, d2 = 6;
	int a1 = 0, a2 = 0;
	ulong m1 = 0, m2 = 0;
	bool esc1 = is_escaped_root_p(search, b, player, &a1, &d1, &m1);
	bool esc2 = is_escaped_root_o(search, b, player, &a2, &d2, &m2);

	if (!esc1 && !esc2)
		return false;
	
	if (player == 1) {
		if (d1 <= d2) {
			*winner = 1;
			*action = a1;
			*escaped_mask = m1;
			*escaped_depth = d1 * 2;
		}
		else {
			*winner = -1;
			*escaped_mask = m2;
			*escaped_depth = d2 * 2 + 1;
		}
	}
	else {
		if (d2 <= d1) {
			*winner = -1;
			*action = a2;
			*escaped_mask = m2;
			*escaped_depth = d2 * 2;
		}
		else {
			*winner = 1;
			*escaped_mask = m1;
			*escaped_depth = d1 * 2 + 1;
		}
	}

	return true;
}

bool is_done_by_captureing(SearchParam* search, Board* b, int* winner, int* type, ulong* escaped_mask) {
	if (b->p_b == 0) {
		*winner = -1;
		*type = WIN_BLUE4;
		return true;
	}

	if (b->p_r == 0) {
		*winner = 1;
		*type = WIN_RED4;
		return true;
	}

	if (popcount(b->o_b | b->o_r | b->o_u) <= 1) {
		*winner = 0;
		*type = END_CAP7;
		return true;
	}

	*winner = 0;
	*type = WIN_NONE;
	return false;
}

const int EVAL_OFFSET = 100; 

SolveResult solve(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	int winner = 0;
	int type = WIN_NONE;
	ulong captured_mask = 0;

	if(is_done_by_captureing(search, board, &winner, &type, &captured_mask) && type == WIN_RED4) {
		switch(type) {
			case WIN_BLUE4:
				return {0, 0, -1};

			case WIN_RED4:
				return {winner * (EVAL_OFFSET + depth), 0, -1};

			case END_CAP7:
				return {winner * (EVAL_OFFSET + depth), 0, 0};
		}
	}

	int action = 0;
	int escaped_depth = 0;
	ulong escaped_mask = 0;

	if(is_escaped_root(search, board, player, &winner, &action, &escaped_depth, &escaped_mask)) {
		if(winner == -1) {
			return {winner * (EVAL_OFFSET + depth - escaped_depth), escaped_mask, 1};
		}

		int n_cap_red = search->init_cap_o_r_cnt + search->init_o_r_cnt + search->init_o_u_cnt - popcount(board->o_r | board->o_u);
		if (n_cap_red < 4) {
			return {winner * (EVAL_OFFSET + depth - escaped_depth), escaped_mask, 1};
		}
	}

	if (depth <= 0)
		return SOLVE_RESULT_NONE;

	vector<ulong> moves = get_moves(board, player);

	vector<SolveResult> results;

	Board* next_board = new Board(0, 0, 0, 0, 0);
	SolveResult max_result = {-1000000 * player, 0, -1};
	ulong move, next;

	for (int d = 0; d < 4; d++) {
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0) {
			moves_d = moves_d ^ move;

			step(board, next_board, move, d);

			if(depth == 5) {
				//test_f = d == 1 && tzcnt(move) == 26;
			}

			SolveResult result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

			if(depth == 5 && test_f) {
				cout << "alpha, beta: " << alpha << ", " << beta << endl;
				cout << "depth: " << depth << endl;
				cout << "player: " << player << endl;
				cout << "eval: " << result.eval << endl;
				cout << "cause: " << tzcnt(result.cause_piece_mask) << ", " << result.cause_piece_color << endl;
				cout << b_to_string(next_board) << endl;
			}

			if (test_f && depth == 4) {
				cout << "alpha, beta: " << alpha << ", " << beta << endl;
				cout << "depth: " << depth << endl;
				cout << "player: " << player << endl;
				cout << "eval: " << result.eval << endl;
				cout << "cause: " << tzcnt(result.cause_piece_mask) << ", " << result.cause_piece_color << endl;
				cout << b_to_string(next_board) << endl;
			}

			if (result.eval == -(EVAL_OFFSET + depth - 1) && result.cause_piece_mask == 0 && result.cause_piece_color == 0) {
				result.cause_piece_mask = move;
			}

			next = shift_left(move, DIRECTIONS[d]);
			if((next & result.cause_piece_mask) != 0) {
				result.cause_piece_mask = move;
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

	if (test_f && depth == 4) {
		cout << "max eval: " << max_result.eval * player << endl;
	}

	delete next_board;

	if (max_result.eval * player >= 0 || max_result.eval > 0)
		return max_result;
	
	vector<int> colors(36);

	for(SolveResult result: results) {
		int pos = tzcnt(result.cause_piece_mask);
		int color = result.cause_piece_color;

		if (colors[pos] != 0 && colors[pos] != color + 1) {
			return SOLVE_RESULT_NONE;
		}

		colors[pos] = color + 1;
	}
	return max_result;
}

int solve_root(SearchParam* search, Board* board, int alpha, int beta, int player, int depth, int* max_action, ulong* escaped_mask) {
	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0, 0, 0, 0);
	int max_e = -1000000;
	ulong move, next;

	int winner = 0;
	int action = 0;
	int escaped_root_depth = 0;
	int escaped_root_e = 0;

	if (is_escaped_root(search, board, player, &winner, max_action, &escaped_root_depth, escaped_mask)) {
		escaped_root_e = winner * (EVAL_OFFSET + depth - escaped_root_depth);

		if(winner != 0 && escaped_root_depth == 0)
			return escaped_root_e;
	}

	// cout << "root: " << winner << ", " << escaped_root_depth << endl;

	vector<SolveResult> results;
	SolveResult result = {0, 0, -1};

	for (int d = 0; d < 4; d++) {
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0) {
			moves_d = moves_d ^ move;

			step(board, next_board, move, d);

			ulong escaped_mask_i = 0;
			int type = 0;
			if(is_done_by_captureing(search, next_board, &winner, &type, &escaped_mask_i)) {
				switch(type) {
					case WIN_BLUE4:
						continue;

					case WIN_RED4:
						result = {winner * (EVAL_OFFSET + depth), 0, -1};

					case END_CAP7:
						result = {winner * (EVAL_OFFSET + depth), move, -0};
				}
			}
			else {
				int escaped_root_e_i = 0;

				if (is_escaped_root(search, next_board, -player, &winner, &action, &escaped_root_depth, &escaped_mask_i)) {
					if (winner == -1 || search->n_cap_red_in_worst_case(next_board) < 4)
						escaped_root_e_i = winner * (EVAL_OFFSET + depth - 1 - escaped_root_depth);
				}

				result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

				if(abs(escaped_root_e_i) > abs(result.eval)) {
					result.eval = escaped_root_e_i;
					result.cause_piece_mask = escaped_mask_i;
					result.cause_piece_color = 1;
				}
				else {
					next = shift_left(move, DIRECTIONS[d]);
					if((next & result.cause_piece_mask) != 0) {
						result.cause_piece_mask = move;
					}
				}
			}

			results.push_back(result);

			if (result.eval * player > max_e) {
				max_e = result.eval * player;

				if(abs(result.eval) > abs(escaped_root_e)) {
					*max_action = tzcnt(move) * 4 + d;
					*escaped_mask = result.cause_piece_mask;
				}
			}
			alpha = max(alpha, result.eval * player);
		}
	}
	delete next_board;

	if (max_e * player >= 0 || max_e > 0)
		return max_e * player;

	vector<int> colors(36);

	for(SolveResult result: results) {
		int pos = tzcnt(result.cause_piece_mask);
		int color = result.cause_piece_color;

		if (colors[pos] != 0 && colors[pos] != color + 1) {
			return 0;
		}
		colors[pos] = color + 1;
	}

	return max_e * player;
}


py::tuple find_checkmate(
	py::array_t<int> pos_p, py::array_t<int> color_p,
	py::array_t<int> pos_o, py::array_t<int> color_o,
	int turn_player, int player, int depth) {

	ulong p_b = 0;
	ulong p_r = 0;
	ulong o_b = 0;
	ulong o_r = 0;
	ulong o_u = 0;

	int init_cap_o_b_cnt = 0;
	int init_cap_o_r_cnt = 0;

	for (int i = 0; i < 8; i++) {
		int p_i = *pos_p.data(i);
		int c_p_i = *color_p.data(i);

		if (p_i >= 0) {
			if (c_p_i == 0)
				p_r |= x_to_bit(p_i);
			else
				p_b |= x_to_bit(p_i);
		}

		int o_i = *pos_o.data(i);
		int c_o_i = *color_o.data(i);

		if (o_i >= 0) {
			if (c_o_i == 0) 
				o_r |= x_to_bit(o_i);
			else if (c_o_i == 1) 
				o_b |= x_to_bit(o_i);
			else 
				o_u |= x_to_bit(o_i);
		}
		else {
			if (c_o_i == 0) 
				init_cap_o_r_cnt++;
			else 
				init_cap_o_b_cnt++;
		}
	}

	SearchParam search = {player, o_b, o_r, o_u, init_cap_o_b_cnt, init_cap_o_r_cnt};

	Board* board = new Board(p_b, p_r, o_b, o_r, o_u);

	int max_action;
	ulong escaped_mask;

	int e = solve_root(&search, board, -1000, 1000, turn_player, depth, &max_action, &escaped_mask);

	delete board;

	if (e == 0) {
		return py::make_tuple(-1, e, -1);
	}

	int pos = max_action / 4;
	int max_d = max_action % 4;
	int escaped_pos = tzcnt(escaped_mask);

	int move_id = -1;
	int escaped_id = -1;

	for (int i = 0; i < 8; i++) {
		int p_i;
		if(turn_player == 1)
			p_i = *pos_p.data(i);
		else
			p_i = *pos_o.data(i);

		if (p_i == pos) {
			move_id = i;
		}

		if (*pos_o.data(i) == escaped_pos) {
			escaped_id = i;
		}
	}

	int action = move_id * 4 + max_d;
	return py::make_tuple(action, e, escaped_id);
}

PYBIND11_MODULE(checkmate_lib, m) {
	m.def("find_checkmate", &find_checkmate);
}
