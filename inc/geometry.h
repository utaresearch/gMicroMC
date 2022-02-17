#ifndef GEOMETRY_H
#define GEOMETRY_H

#include<map>
#include "global.h"

class Geometry
{
public:
    std::string name;
    float p[6][10]={0}; // for each row, ten parameters defined a volume
    // p[0]x^2+p[1]y^2+p[2]z^2+p[3]xy+p[4]yz+p[5]zx+p[6]x+p[7]y+p[8]z+p[9]<=0;

    iniAnyShape(int num, float *parray);
    iniSphere(float xc, float yc, float zc, float r);
    iniBox(float xc, float yc, float zc, float sx, float sy, float sz);
    // plane is same to box with sx/sy/sz=0;
    iniCylinder(float xc, float yc, float zc, float r, float height);
    // circle is same to cylinder with height=0

    
}

#endif
