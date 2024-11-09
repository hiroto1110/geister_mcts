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
	ulong p, o;

	Board(ulong p, ulong o) {
		this->p = p;
		this->o = o;
	}
};

void invert(Board* src, Board* dst) {
	dst->p = src->o;
	dst->o = src->p;
}

string b_to_string(Board* b) {
	const string line = "+---+---+---+---+---+---+";
	string s = line + "\r\n";

	for (int i = 0; i < 36; i++) {
		if (i % 6 == 0 && i != 0) {
			s += "|\r\n" + line + "\r\n";
		}

		if (b->p & x_to_bit(i)) {
			s += "| X ";
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
		return GetMoves(b->p);
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

	if ((b->p & move) != 0) {
		next_board->p = b->p ^ diff;
		next_board->o = b->o & ~next;
		return;
	}
	else if ((b->o & move) != 0) {
		next_board->p = b->p & ~next;
		next_board->o = b->o ^ diff;
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
	int init_cap_p_b_cnt, init_cap_p_r_cnt;
	int init_cap_o_b_cnt, init_cap_o_r_cnt;
	int init_p_cnt, init_o_cnt;

	SearchParam() {
	}

	SearchParam(int root_player, ulong p, ulong o, int cap_p_b, int cap_p_r, int cap_o_b, int cap_o_r) {
		this->root_player = root_player;

		this->init_p_cnt = popcount(p);
		this->init_o_cnt = popcount(o);
		this->init_cap_p_b_cnt = cap_p_b;
		this->init_cap_p_r_cnt = cap_p_r;
		this->init_cap_o_b_cnt = cap_o_b;
		this->init_cap_o_r_cnt = cap_o_r;
	}

	int n_cap_blue_in_best_case(Board* b, int player) {
		if(player == 1)
			return this->init_cap_o_b_cnt + this->init_o_cnt - popcount(b->o);
		else
			return this->init_cap_p_b_cnt + this->init_p_cnt - popcount(b->p);
	}

	int n_cap_red_in_worst_case(Board* b, int player) {
		if(player == 1)
			return this->init_cap_o_r_cnt + this->init_o_cnt - popcount(b->o);
		else
			return this->init_cap_p_r_cnt + this->init_p_cnt - popcount(b->p);
	}
};

bool test_f = true;

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

bool is_escaped_root(Board* b, int player, int root_player, int i, int* distance, ulong* escaped_mask, bool* is_tension_state) {
	// vector<ulong>* masks = (root_player == 1) ? &(DISTANCE_MASKS_O.at(i)) : &(DISTANCE_MASKS_P.at(i));
	vector<ulong> masks = (root_player == 1) ? DISTANCE_MASKS_O.at(i) : DISTANCE_MASKS_P.at(i);

	int distance_p = calc_min_distance(&masks, b->p);
	int distance_o = calc_min_distance(&masks, b->o);

	cout << distance_p << ", " << distance_o << endl;

	int offset = player == -1 ? 0 : -1;

	if(distance_o == distance_p + offset) {
		*is_tension_state = true;
		return false;
	}

	if(distance_o < distance_p + offset) {
		*distance = distance_o;

		ulong mask = b->o & masks.at(distance_o);
		*escaped_mask = first_bit(mask);

		return true;
	}
	return false;
}

bool is_escaped_root(Board* b, int player, int root_player, int* distance, ulong* escaped_mask, bool* is_tension_state) {
	int d1 = 6, d2 = 6;
	ulong m1 = 0, m2 = 0;
	bool t1 = false, t2 = false;
	bool esc1 = is_escaped_root(b, player, root_player, 0, &d1, &m1, &t1);
	bool esc2 = is_escaped_root(b, player, root_player, 1, &d2, &m2, &t2);

	*is_tension_state = t1 || t2;

	if (!esc1 && !esc2) {
		return false;
	}

	if(d1 < d2) {
		*distance = d1;
		*escaped_mask = m1;
	}
	else {
		*distance = d2;
		*escaped_mask = m2;
	}
	return true;
}

bool is_escaped_root(SearchParam* search, Board* b, int player, int* winner, int* escaped_depth, ulong* escaped_mask) {
	int d1 = 6, d2 = 6;
	ulong m1 = 0, m2 = 0;
	bool t1 = false, t2 = false;

	Board* b_org = b;
	Board* b_inv = new Board(b_org->o, b_org->p);

	bool esc1 = is_escaped_root(b_inv, -player, -search->root_player, &d1, &m1, &t1);
	bool esc2 = is_escaped_root(b_org, player, search->root_player, &d2, &m2, &t2);

	cout << "e: " << esc1 << ", " << esc2 << endl;
	cout << "d: " << d1 << ", " << d2 << endl;
	cout << "t: " << t1 << ", " << t2 << endl;

	delete b_inv;

	if (!esc1 && !esc2)
		return false;

	if (t1 || t2)
		return false;

	if (player == 1) {
		if (d1 <= d2) {
			*winner = 1;
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

bool is_done_by_captureing(Board* b) {
	if ((popcount(b->o) <= 1) || (popcount(b->p) <= 1)) {
		return true;
	}

	return false;
}

const int EVAL_OFFSET = 100; 

SolveResult solve(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	if(is_done_by_captureing(board)) {
		return {0, 0, 0};
	}

	int winner = 0;
	int escaped_depth = 0;
	ulong escaped_mask = 0;

	if(is_escaped_root(search, board, player, &winner, &escaped_depth, &escaped_mask)) {
		int n_cap_red;

		if(winner == 1)
			n_cap_red = search->init_cap_o_r_cnt + search->init_o_cnt - popcount(board->o);
		else
			n_cap_red = search->init_cap_p_r_cnt + search->init_p_cnt - popcount(board->p);

		if(n_cap_red < 4) {
			return {winner * (EVAL_OFFSET + depth - escaped_depth), escaped_mask, 1};
		}
	}

	if (depth <= 0)
		return SOLVE_RESULT_NONE;

	vector<ulong> moves = get_moves(board, player);

	vector<SolveResult> results;

	Board* next_board = new Board(0, 0);
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

	if (max_result.eval == 0)
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

int solve_root(SearchParam* search, Board* board, int alpha, int beta, int player, int depth) {
	vector<ulong> moves = get_moves(board, player);

	Board* next_board = new Board(0, 0);
	int max_e = -1000000;
	ulong move, next;

	int winner = 0;
	int escaped_root_depth = 0;
	ulong escaped_mask = 0;
	int escaped_root_e = 0;

	if (is_escaped_root(search, board, player, &winner, &escaped_root_depth, &escaped_mask)) {
		cout << "root: " << winner << ", " << escaped_root_depth << endl;
		escaped_root_e = winner * (EVAL_OFFSET + depth - escaped_root_depth);

		if(winner != 0 && escaped_root_depth == 0)
			return escaped_root_e;
	}

	cout << "root: " << winner << ", " << escaped_root_depth << endl;

	vector<SolveResult> results;
	SolveResult result = {0, 0, -1};

	for (int d = 0; d < 4; d++) {
		ulong moves_d = moves[d];

		while ((move = first_bit(moves_d)) != 0) {
			moves_d = moves_d ^ move;

			step(board, next_board, move, d);

			ulong escaped_mask_i = 0;
			if(is_done_by_captureing(next_board)) {
				result = {0, move, -0};
			}
			else {
				int escaped_root_e_i = 0;

				if (is_escaped_root(search, next_board, -player, &winner, &escaped_root_depth, &escaped_mask)) {
					if (search->n_cap_red_in_worst_case(next_board, winner) < 4)
						escaped_root_e_i = winner * (EVAL_OFFSET + depth - 1 - escaped_root_depth);
				}

				result = solve(search, next_board, -beta, -alpha, -player, depth - 1);

				/*cout << "alpha, beta: " << alpha << ", " << beta << endl;
				cout << "player: " << player << endl;
				cout << "eval: " << result.eval << endl;
				cout << "cause: " << tzcnt(result.cause_piece_mask) << ", " << result.cause_piece_color << endl;
				cout << b_to_string(next_board) << endl;*/

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
			}
			alpha = max(alpha, result.eval * player);
		}
	}
	delete next_board;

	return max_e;

	if (max_e == 0)
		return 0;

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

int main() {
	ulong p = 0b011110011110ULL;

	ulong o = 0b000000000000000000000000011110011110ULL;
	
	int init_cap_p_b_cnt = 0;
	int init_cap_p_r_cnt = 0;
	int init_cap_o_b_cnt = 0;
	int init_cap_o_r_cnt = 0;

	Board* board = new Board(p, o);
	SearchParam search = {1, p, o, init_cap_p_b_cnt, init_cap_p_r_cnt, init_cap_o_b_cnt, init_cap_o_r_cnt};

	int e = solve_root(&search, board, -1000, 1000, 1, 5);

	cout << e << endl;
}


int find_checkmate(
	py::array_t<int> pos_p, py::array_t<int> color_p,
	py::array_t<int> pos_o, py::array_t<int> color_o,
	int turn_player, int player, int depth
) {

	ulong p = 0;
	ulong o = 0;

	int init_cap_p_b_cnt = 0;
	int init_cap_p_r_cnt = 0;
	int init_cap_o_b_cnt = 0;
	int init_cap_o_r_cnt = 0;

	for (int i = 0; i < 8; i++) {
		int p_i = *pos_p.data(i);
		int c_p_i = *color_p.data(i);

		if (p_i >= 0) {
			p |= x_to_bit(p_i);
		}
		else {
			if (c_p_i == 0)
				init_cap_p_r_cnt++;
			else
				init_cap_p_b_cnt++;
		}

		int o_i = *pos_o.data(i);
		int c_o_i = *color_o.data(i);

		if (o_i >= 0) {
			o |= x_to_bit(o_i);
		}
		else {
			if (c_o_i == 0) 
				init_cap_o_r_cnt++;
			else 
				init_cap_o_b_cnt++;
		}
	}

	SearchParam search = {player, p, o, init_cap_p_b_cnt, init_cap_p_r_cnt, init_cap_o_b_cnt, init_cap_o_r_cnt};

	Board* board = new Board(p, o);

	int e = solve_root(&search, board, -1000, 1000, turn_player, depth);

	delete board;

	return e;
}

PYBIND11_MODULE(checkmate_u_lib, m) {
	m.def("find_checkmate", &find_checkmate);
}