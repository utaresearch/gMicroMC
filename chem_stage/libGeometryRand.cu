#ifndef __LIBGEOMETRY_RAND_CU__
#define __LIBGEOMETRY_RAND_CU__

__device__ void rotate(float3 &direction, float costh, float phi)
/*******************************************************************
c*    Rotates a vector; the rotation is specified by giving        *
c*    the polar and azimuthal angles in the "self-frame", as       *
c*    determined by the vector to be rotated.                      *
c*                                                                 *
c*    Input:                                                       *
c*      float3 direction =(u,v,w) -> input vector (=d) in the lab. frame             *
c*      costh -> cos(theta), angle between d before and after turn *
c*      phi -> azimuthal angle (rad) turned by d in its self-frame *
c*    Output:                                                      *
c*      float3 direction=(u,v,w) -> rotated vector components in the lab. frame     *
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
        float rho2,sinphi,cosphi,sthrho,urho,vrho,sinth;

        rho2 = direction.x * direction.x + direction.y * direction.y;
       
        sinphi = __sinf(phi);
        cosphi = __cosf(phi);
//      Case z' not= z:

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
//      2 especial cases when z'=z or z'=-z:
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

void inirngG(int value)
{
//  initialize rand seeds at CPU
    printf("\nStart initialize random numbers\n");
    if(value == 0)
    {
        srand( (unsigned int)time(NULL) );
    }
    else 
    {
        srand ( value );
    }
    iseed1_h = (int*) malloc(sizeof(int)*MAXNUMPAR2);
    if(iseed1_h==NULL) printf("MALLOC error\n");
//  generate randseed at CPU
    for(int i = 0; i < MAXNUMPAR2; i++)
    {
        iseed1_h[i] = rand();
    }
    CUDA_CALL(cudaMalloc((void **) &iseed1,sizeof(int)*MAXNUMPAR2));
//  copy to GPU 
    CUDA_CALL(cudaMemcpy(iseed1, iseed1_h, sizeof(int)*MAXNUMPAR2, cudaMemcpyHostToDevice));

    int nblocks;
    nblocks = 1 + (MAXNUMPAR2 - 1)/NTHREAD_PER_BLOCK_PAR ;
    setupcuseed<<<nblocks, NTHREAD_PER_BLOCK_PAR>>>(iseed1);
    cudaDeviceSynchronize();

    //revised at Feb 21
    free(iseed1_h);
    cudaFree(iseed1);
}


__global__ void setupcuseed(int* iseed1)
//  setup random seeds
{
    const int id = blockIdx.x*blockDim.x + threadIdx.x;
//      obtain current id on thread
    if( id < MAXNUMPAR2)
    {
        curand_init(iseed1[id], id, 0, &cuseed[id]);
    }
    if(id<5) printf("the first 5 random seeds are %u\n", cuseed[id]);
}
#endif
