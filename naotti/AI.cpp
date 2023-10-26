﻿#include <iostream>
#include <ostream>
#include "Game.h"
#include "Search.h"
#include "KanzenSearch.h"
#include <vector>
#include <map>
#include <ctime>
#include <algorithm>
#include <cassert>
#include <tuple>
#include <functional>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using namespace std;
using namespace Game_;

//逃げ, 追いかけの判定
int dy[4] = { -1, 0, 1, 0 };
int dx[4] = { 0, 1, 0, -1 };
bool isNige(char board[][6], MoveCommand te) {
	int ny = te.y + dy[te.dir];
	int nx = te.x + dx[te.dir];
	if (ny < 0 || ny >= 6 || nx < 0 || nx >= 6) return false;	//脱出手は「逃げ」ではない
	if (board[ny][nx] == 'u') return false;			//駒を取る手は「逃げ」ではない

	int i;
	for (i = 0; i < 4; i++) {
		int y = te.y + dy[i];
		int x = te.x + dx[i];
		if (0 <= y && y < 6 && 0 <= x && x < 6 && board[y][x] == 'u') {
			break;
		}
	}
	if (i == 4) { return false; }	//動かす駒の「動かす前のマス」と隣接するマスに相手の駒がなかったら、「逃げ」ではない

	for (i = 0; i < 4; i++) {
		int y = ny + dy[i];
		int x = nx + dx[i];
		if (0 <= y && y < 6 && 0 <= x && x < 6 && board[y][x] == 'u') {
			break;
		}
	}
	if (i < 4) { return false; }	//動かす駒の「動かした後のマス」と隣接するマスに相手の駒があったら、「逃げ」ではない
	return true;	//逃げである
}

bool isOikake(char board[][6], MoveCommand te) {
	int ny = te.y + dy[te.dir];
	int nx = te.x + dx[te.dir];
	if (ny < 0 || ny >= 6 || nx < 0 || nx >= 6) return false;	//脱出手は「追いかけ」ではない
	if (board[ny][nx] == 'u') return false;			//駒を取る手は「追いかけ」ではない

	int i;
	for (i = 0; i < 4; i++) {
		int y = te.y + dy[i];
		int x = te.x + dx[i];
		if (0 <= y && y < 6 && 0 <= x && x < 6 && board[y][x] == 'u') {
			break;
		}
	}
	if (i < 4) { return false; }	//動かす駒の「動かす前のマス」と隣接するマスに相手の駒があったら「追いかけ」ではない

	for (i = 0; i < 4; i++) {
		int y = ny + dy[i];
		int x = nx + dx[i];
		if (0 <= y && y < 6 && 0 <= x && x < 6 && board[y][x] == 'u') {
			break;
		}
	}
	if (i == 4) { return false; }	//動かす駒の「動かした後のマス」と隣接するマスに相手の駒がなかったら「追いかけ」ではない
	return true;	//「追いかけ」である
}

bool printLog = false;

//逃げ, 追いかけの回数
int nigeR, nigeB, oikakeR, oikakeB;
void AddNigeR() { nigeR++; if (printLog) cout << "Add nigeR" << endl; }
void AddNigeB() { nigeB++; if (printLog) cout << "Add nigeB" << endl; }
void AddOikakeR() { oikakeR++; if (printLog) cout << "Add oikakeR" << endl; }
void AddOikakeB() { oikakeB++; if (printLog) cout << "Add oikakeB" << endl; }

//BEGIN: 赤度を推定する系統
namespace red {
	int histCnt;
	char hist[350][6][6];	//R, B, u
	double eval[350][6][6];	//赤度

	void moveHist(char prev[6][6], char now[6][6], MoveCommand mv);
	void moveEval(double prev[6][6], double now[6][6], MoveCommand mv);
	MoveCommand detectMove(char prev[6][6], char now[6][6]);
	int toDir(int y, int x, int ny, int nx);

	//試合終了後に呼び出す
	void saveGame() {
		int i, j, k;
		for (i = 0; i < histCnt; i++) {
			for (j = 0; j < 6; j++) {
				for (k = 0; k < 6; k++) {
					cout << hist[i][j][k];
				}
				cout << endl;
			}
			cout << endl;

			for (j = 0; j < 6; j++) {
				for (k = 0; k < 6; k++) {
					cout << eval[i][j][k] << " ";
				}
				cout << endl;
			}
			cout << endl;
		}
	}

	//試合開始時に呼び出す
	void initGame(char initBoard[6][6]) {
		int i, j;

		histCnt = 0;
		for (i = 0; i < 6; i++) {
			for (j = 0; j < 6; j++) {
				hist[0][i][j] = initBoard[i][j];
				eval[0][i][j] = 0;
			}
		}
		histCnt++;
	}

	//自分が手を打ったときに呼び出す
	void myMove(MoveCommand mv) {
		moveHist(hist[histCnt - 1], hist[histCnt], mv);
		moveEval(eval[histCnt - 1], eval[histCnt], mv);
		histCnt++;
	}

	//2手目以降の自分手番の最初に呼び出す。
	void myTurn(char board[6][6]) {
		int i, j;

		for (i = 0; i < 6; i++)
			for (j = 0; j < 6; j++)
				hist[histCnt][i][j] = board[i][j];
		histCnt++;

		MoveCommand mv = detectMove(hist[histCnt - 2], hist[histCnt - 1]);
		moveEval(eval[histCnt - 2], eval[histCnt - 1], mv);
		int ny = mv.y + dy[mv.dir];
		int nx = mv.x + dx[mv.dir];

		char block[6][6];
		int prevMyRed = 0;
		for (i = 0; i < 6; i++) {
			for (j = 0; j < 6; j++) {
				char c = hist[histCnt - 2][i][j];
				if (c == 'R' || c == 'B') block[i][j] = 'u';
				else block[i][j] = '.';
				if (c == 'R') prevMyRed++;
			}
		}

		int weightOikake = 5;
		int weightOikakePinti = 1;	//相手がピンチなとき、わけわからん行動しそうなので、推定の信頼を低めに
		if (isOikake(block, mv)) {
			if(printLog)
				cerr << "Enemy(" << ny << ", " << nx << ") is Oikaked." << endl;
			if (prevMyRed == 1) {
				if(printLog)
					cerr << "相手はピンチだった" << endl;
				eval[histCnt - 1][ny][nx] += weightOikakePinti;
			}
			else {
				if(printLog)
					cerr << "相手は余裕だった" << endl;
				eval[histCnt - 1][ny][nx] += weightOikake;
			}
		}

		//相手が動かした駒を見て、それが青だったら自分がどう頑張っても必ず負けるとき、赤だと思って
		//見捨てる。
		//本当はちゃんと「相手側の必勝手探索」を実装したかったけど、時間がないので手抜きで。
		int weightHairi = 1000;
		if ((mv.y == 5 && mv.x == 0) || (mv.y == 5 && mv.x == 5)) {	//こいつ脱出しなかったから赤だゾ
			if(printLog)
				cerr << "Enemy(" << ny << ", " << nx << ") は脱出しなかったから赤だゾ" << endl;
			eval[histCnt - 1][ny][nx] += weightHairi;
		}

		//今脱出口にある相手駒が、直前に動かしてきたものでなければ、赤
		if (hist[histCnt - 1][5][0] == 'u' && !(ny == 5 && nx == 0)) {
			if(printLog)
				cerr << "Enemy(" << 5 << ", " << 0 << ") は留まってるから赤だゾ" << endl;
			eval[histCnt - 1][5][0] += weightHairi;
		}
		if (hist[histCnt - 1][5][5] == 'u' && !(ny == 5 && nx == 5)) {
			if(printLog)
				cerr << "Enemy(" << 5 << ", " << 5 << ") は留まってるから赤だゾ" << endl;
			eval[histCnt - 1][5][5] += weightHairi;
		}
	}

	//myMoveとかmyTurnとかを呼び出した直後に呼び出したい。
	//赤度evalが閾値以上になった赤の現在位置を、赤度が大きいものからリストアップ
	int listUpRed(int posY[], int posX[], int X) {
		int i, j;

		typedef tuple<double, int, int> T;
		vector<T> vec;

		for (i = 0; i < 6; i++) {
			for (j = 0; j < 6; j++) {
				if (eval[histCnt - 1][i][j] >= X) {
					vec.push_back(T(eval[histCnt - 1][i][j], i, j));
				}
			}
		}

		sort(vec.begin(), vec.end(), greater<T>());
		for (i = 0; i < vec.size(); i++) {
			posY[i] = get<1>(vec[i]);
			posX[i] = get<2>(vec[i]);
		}
		return vec.size();
	}

	void moveHist(char prev[6][6], char now[6][6], MoveCommand mv) {
		int y = mv.y;
		int x = mv.x;
		int ny = mv.y + dy[mv.dir];
		int nx = mv.x + dx[mv.dir];
		int i, j;

		for (i = 0; i < 6; i++)
			for (j = 0; j < 6; j++)
				now[i][j] = prev[i][j];

		char color = prev[y][x];	//R, B, u
		now[ny][nx] = color;
		now[y][x] = '.';
	}

	void moveEval(double prev[6][6], double now[6][6], MoveCommand mv) {
		int y = mv.y;
		int x = mv.x;
		int ny = mv.y + dy[mv.dir];
		int nx = mv.x + dx[mv.dir];
		int i, j;

		for (i = 0; i < 6; i++)
			for (j = 0; j < 6; j++)
				now[i][j] = prev[i][j];

		double tmp = prev[y][x];
		now[ny][nx] = tmp;
		now[y][x] = 0;
	}

	MoveCommand detectMove(char prev[6][6], char now[6][6]) {
		int i, j;
		int cnt = 0;
		int posY[2], posX[2];

		for (i = 0; i < 6; i++) {
			for (j = 0; j < 6; j++) {
				if (prev[i][j] != now[i][j]) {
					posY[cnt] = i;
					posX[cnt] = j;
					cnt++;
				}
			}
		}
		if(cnt != 2) {
			cout << "test: " << cnt << endl;
			throw 1;
		}
		assert(cnt == 2);

		int y, x, ny, nx;
		if (now[posY[0]][posX[0]] == '.') {
			y = posY[0];
			x = posX[0];
			ny = posY[1];
			nx = posX[1];
		}
		else {
			y = posY[1];
			x = posX[1];
			ny = posY[0];
			nx = posX[0];
		}

		int dir = toDir(y, x, ny, nx);
		return MoveCommand(y, x, dir);
	}

	int toDir(int y, int x, int ny, int nx) {
		int dir;
		for (dir = 0; dir < 4; dir++) {
			if (dy[dir] == ny - y && dx[dir] == nx - x) {
				break;
			}
		}
		assert(dir <= 3);
		return dir;
	}
}
//END

//指し手を決める (Search.hは, (盤面, pnum)が同じなら必ず同じ手を返すアルゴリズムなので、メモ化しても問題なし）
Search searchObj;
KanzenSearch kanzenObj;
clock_t sumThinkTime = 0;
int maxDepth;		//探索の深さ(推奨：5～6）

//それ以外全部青！
pair<MoveCommand, int> thinkKanzen(int X) {
	int i, j;

	int posY[8], posX[8];
	int cnt = red::listUpRed(posY, posX, X);
	if (cnt == 0) return pair<MoveCommand, int>(MoveCommand(-1, -1, -1), 0);

	string s;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			if (i == posY[0] && j == posX[0]) {
				s += "r";
			}
			else if (board[i][j] == 'u') {
				s += "b";
			}
			else {
				s += board[i][j];
			}
		}
	}
	return kanzenObj.think(s, maxDepth);
}

//紫駒
pair<MoveCommand, int> thinkPurple() {
	int i, j;

	string s;
	for (i = 0; i < 6; i++) {
		for (j = 0; j < 6; j++) {
			s += board[i][j];
		}
	}
	int pnum = Game_::uNum - Game_::rNum;
	BitBoard bb;
	bb.toBitBoard(s);
	return searchObj.think(bb, pnum, maxDepth);
}

//手を決める
pair<MoveCommand, int> thinkMove() {
	pair<MoveCommand, int> resK, resP;
	resK = thinkKanzen(1000);
	if (resK.first.y >= 0) return resK;

	if (Game_::rNum >= 1) {
		resP = thinkPurple();
		if (resP.second >= -searchObj.INF / 2) return resP;
	}

	resK = thinkKanzen(1000);
	if (resK.first.y >= 0) return resK;
	return resP;
}

//手を決めるのと、いろんな処理
string solve(int turnCnt) {
	//時間計測開始
	clock_t startTime = clock();

	//赤らしさの更新
	if (turnCnt == 0) {
		red::initGame(board);
	}
	else {
		red::myTurn(board);

		if(printLog) {
			int i, j;
			cerr << "赤度" << endl;
			for (i = 0; i < 6; i++) {
				for (j = 0; j < 6; j++) {
					cerr << red::eval[red::histCnt - 1][i][j] << " ";
				}
				cerr << endl;
			}
		}
	}

	//手を決める
	pair<MoveCommand, int> res = thinkMove();
	MoveCommand te = res.first;

	//思考時間
	sumThinkTime += clock() - startTime;

	//情報の更新
	if (isNige(board, te)) { if (board[te.y][te.x] == 'R') AddNigeR(); else if (board[te.y][te.x] == 'B') AddNigeB(); else assert(0); }
	if (isOikake(board, te)) { if (board[te.y][te.x] == 'R') AddOikakeR(); else if (board[te.y][te.x] == 'B') AddOikakeB(); else assert(0); }
	red::myMove(te);

	//表示
	if(printLog)
		cerr << "選択手(" << te.y << ", " << te.x << ", " << te.dir << "), 評価値 = " << res.second << endl;
	return move(te.y, te.x, te.dir);
}

void initGame(int depth, bool print) {
	maxDepth = depth;
	printLog = print;

	bb::weight1 = 1000;
	bb::weight2 = 1;
	kbb::weight1 = 1000;
	kbb::weight2 = 1;

	bb::prepare();
	kbb::prepare();

	nigeR = nigeB = oikakeR = oikakeB = 0;	//逃げ回数, 追いかけ回数の初期化
}


PYBIND11_MODULE(naotti2020, m) {
	m.def("solve", &solve);
	m.def("recvBoard", &recvBoard);
	m.def("initGame", &initGame);
}
