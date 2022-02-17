#include "globalKernel.cuh"
__device__ curandState *cuseed;

__device__ void rotate(gFloat *u, gFloat *v, gFloat *w, gFloat costh, gFloat phi)
/*******************************************************************
c*    Rotates a vector; the rotation is specified by giving        *
c*    the polar and azimuthal angles in the "self-frame", as       *
c*    determined by the vector to be rotated.                      *
c*                                                                 *
c*    Input:                                                       *
c*      (u,v,w) -> input vector (=d) in the lab. frame             *
c*      costh -> cos(theta), angle between d before and after turn *
c*      phi -> azimuthal angle (rad) turned by d in its self-frame *
c*    Output:                                                      *
c*      (u,v,w) -> rotated vector components in the lab. frame     *
c*    Comments:                                                    *
c*      -> (u,v,w) should have norm=1 on input; if not, it is      *
c*         renormalized on output, provided norm>0.                *
c*      -> The algorithm is based on considering the turned vector *
c*         d' expressed in the self-frame S',                      *
c*           d' = (sin(th)cos(ph), sin(th)sin(ph), cos(th))        *
c*         and then apply a change of frame from S' to the lab     *
c*         frame. S' is defined as having its z' axis coincident   *
c*         with d, its y' axis perpendicular to z and z' and its   *
c*         x' axis equal to y'*z'. The matrix of the change is then*
c*                   / uv/rho    -v/rho    u \                     *
c*          S ->lab: | vw/rho     u/rho    v |  , rho=(u^2+v^2)^0.5*
c*                   \ -rho       0        w /                     *
c*      -> When rho=0 (w=1 or -1) z and z' are parallel and the y' *
c*         axis cannot be defined in this way. Instead y' is set to*
c*         y and therefore either x'=x (if w=1) or x'=-x (w=-1)    *
c******************************************************************/
{
    float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth,norm;

    rho2 = (*u)*(*u)+(*v)*(*v);
    norm = rho2 + (*w)*(*w);
//      Check normalization:
    if (fabs(norm-1.0) > SZERO)
    {
//      Renormalize:
        norm = 1.0/__fsqrt_rn(norm);
        *u = (*u)*norm;
        *v = (*v)*norm;
        *w = (*w)*norm;
    }

    sinphi = __sinf(phi);
    cosphi = __cosf(phi);
//      Case z' not= z:

	float temp = costh*costh;
    if (rho2 > ZERO)
    {
        if(temp < 1.0f)
			sthrho = __fsqrt_rn((1.00-temp)/rho2);
		else 
			sthrho = 0.0f;

        urho =  (*u)*sthrho;
        vrho =  (*v)*sthrho;
        *u = (*u)*costh - vrho*sinphi + (*w)*urho*cosphi;
        *v = (*v)*costh + urho*sinphi + (*w)*vrho*cosphi;
        *w = (*w)*costh - rho2*sthrho*cosphi;
    }
    else
//      2 especial cases when z'=z or z'=-z:
    {
		if(temp < 1.0f)			
	        sinth = __fsqrt_rn(1.00-temp);
		else
			sinth = 0.0f;

        *v = sinth*sinphi;
        if (*w > 0.0)
        {
            *u = sinth*cosphi;
            *w = costh;
        }
        else
        {
            *u = -sinth*cosphi;
            *w = -costh;
        }
    }
}

__device__ void rotate(float3 &direction, float costh, float phi)
/*******************************************************************
reloaded version of rotate a vector
*******************************************************************/
{
    float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth;

    rho2 = direction.x * direction.x + direction.y * direction.y;
    
    sinphi = __sinf(phi);
    cosphi = __cosf(phi);

	float temp = costh*costh;
    if (rho2 > 1.0e-20f)
    {
        if(temp < 1.0f)
            sthrho = __fsqrt_rn((1.00-temp)/rho2);
    else 
        sthrho = 0.0f;

            urho =  direction.x * sthrho;
            vrho =  direction.y * sthrho;
            direction.x = direction.x * costh - vrho * sinphi + direction.z * urho * cosphi;
            direction.y = direction.y * costh + urho * sinphi + direction.z * vrho * cosphi;
            direction.z = direction.z * costh - rho2 * sthrho * cosphi;
    }
    else
    {
        if(temp < 1.0f)			
                    sinth = __fsqrt_rn(1.00-temp);
        else
            sinth = 0.0f;

        direction.y = sinth*sinphi;
        if (direction.z > 0.0)
        {
                direction.x = sinth*cosphi;
                direction.z = costh;
        }
        else
        {
                direction.x = -sinth*cosphi;
                direction.z = -costh;
        }
    }
}

__global__ void setupcuseed(int num, int* iseed1)
//  setup random seeds
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
    if( id < num)
    {
        curand_init(iseed1[id], id, 0, &cuseed[id]);
    }
    if(id<5) printf("the first 5 random seeds are %u\n", cuseed[id].d);
}


void iniCuseed(int *iseed1)
{
    curandState *tmp;
    CUDA_CALL(cudaMalloc((void**) &tmp, NRAND * sizeof(curandState)));
    CUDA_CALL(cudaMemcpyToSymbol(cuseed, &tmp, sizeof(tmp)));

    int nblocks;
    nblocks = 1 + (NRAND - 1)/NTHREAD_PER_BLOCK;
    setupcuseed<<<nblocks, NTHREAD_PER_BLOCK>>>(NRAND, iseed1);
    CUDA_CALL(cudaDeviceSynchronize());
}

__device__ int applyBoundary(int shape, float3 center, float3 size, float x, float y, float z)
{
    if(shape < 0)
        return 0;
    x -= center.x;
    y -= center.y;
    z -= center.z;

    if(shape == 0)
        return sqrtf(x*x+y*y+z*z)>size.x;
    if(shape == 1)
        return (abs(x)>size.x/2.0f || abs(y)>size.y/2.0f || abs(z)>size.z/2.0f);
    
    return 0;
}

int applyROISearch(int shape, float3 center, float3 size, float x, float y, float z)
{
    if(shape < 0)
        return 0;
    x -= center.x;
    y -= center.y;
    z -= center.z;

    if(shape == 0)
        return sqrtf(x*x+y*y+z*z)<=size.x;
    if(shape == 1)
        return (abs(x)<=size.x/2.0f && abs(y)<=size.y/2.0f && abs(z)<=size.z/2.0f);
    
    return 0;
}