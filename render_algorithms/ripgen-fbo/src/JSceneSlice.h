/*
 * JSceneSlice.h
 *
 *  Created on: Jun 28, 2013
 *      Author: holo
 */

#ifndef JSCENESLICE_H_
#define JSCENESLICE_H_

#include <vector>

class JSceneSlice {
public:
	JSceneSlice();
	virtual ~JSceneSlice();
	inline void insert(float x, float z, char l) {
		pointsX.push_back(x);
		pointsZ.push_back(z);
		pointsL.push_back(l);
	}
public:
	std::vector <float> pointsX;
	std::vector <float> pointsZ;
	std::vector <char> pointsL;
};

#endif /* JSCENESLICE_H_ */
