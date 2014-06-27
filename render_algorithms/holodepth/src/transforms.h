/*
 *  transforms.h
 *  Dblfrustum
 *
 *  Created by HoloVideo Bove Lab on 6/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */
void buildShearOrthographicMatrix(double Xleft, double Xright, double Ybot, double Ytop, 
                                   double n, double f, double q,
                                   float m[16]);
								   
void buildShearOrthographicMatrix2(double Xleft, double Xright, double Ybot, double Ytop, 
                                   double n, double f, double q,
                                   float m[16]);
								   								   
void buildDblPerspectiveMatrix(double Xleft, double Xright, double Ybot, double Ytop, 
                                   double n, double f, double p,
                                   float m[16]);


void buildDblPerspectiveMatrix2(double Xleft, double Xright, double Ybot, double Ytop, 
                                   double n, double f, double p,
                                   float m[16]);


void buildPerspectiveMatrix(double fieldOfView,
                                   double aspectRatio,
                                   double zNear, double zFar,
                                   float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for gluLookAt. */
void buildLookAtMatrix(double eyex, double eyey, double eyez,
                              double centerx, double centery, double centerz,
                              double upx, double upy, double upz,
                              float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glRotatef. */
void makeRotateMatrix(float angle,
                             float ax, float ay, float az,
                             float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glTranslatef. */
void makeTranslateMatrix(float x, float y, float z, float m[16]);

/* Build a row-major (C-style) 4x4 matrix transform based on the
   parameters for glTranslatef. */
void makeScaleMatrix(float x, float y, float z, float m[16]);


/* Simple 4x4 matrix by 4x4 matrix multiply. */
void multMatrix(float dst[16],
                       const float src1[16], const float src2[16]);
					   
/*Invert a row-major (C-style) 4x4 model (trans,rot) matrix */
void transposeMatrix(float *out, const float *m);

/* Invert a row-major (C-style) 4x4 matrix. */
void invertMatrix(float *out, const float *m);

/* Simple 4x4 matrix by 4-component column vector multiply. */
void transform(float dst[4],
                      const float mat[16], const float vec[4]);
