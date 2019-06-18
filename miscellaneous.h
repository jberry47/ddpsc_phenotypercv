/*
 * miscellaneous.h
 *
 *  Created on: Jun 18, 2019
 *      Author: jberry
 */

#ifndef MISCELLANEOUS_H_
#define MISCELLANEOUS_H_

extern int roi_size;
extern Mat src;
extern int counter;

void split(const string& s, char c, vector<string>& v);
void onMouse( int event, int x, int y, int f, void* );


#endif /* MISCELLANEOUS_H_ */
