#ifndef _UTILS_H
#define _UTILS_H

//#define M_PI 3.1415
#define DEG_TO_RAD(x) (2.0*M_PI/360.0*x)

float htonf(float a);
char *get_string(FILE *f);
int str2bool(char *s);
void error(char *errmsg, char *opt);


void multMatrix(float a[3][3], float r[3]);
void transformVect(orientation *o, float n[3], int trans=0);
void inverseTransformVect(orientation *o, float n[3], int trans=0);

#endif
