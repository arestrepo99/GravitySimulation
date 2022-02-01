kernel void applyStep(const float dt,
                global float *position,
                global float *velocity,
                const uint N,
                const float G)
{   
    uint indi = get_global_id(0);
    uint ix = indi*2; uint iy = indi*2+1;
    uint jx; uint jy;  float dir_x; float dir_y;
    float radius; float forceMagnitude; float force_x = 0; float force_y = 0;
    for(uint indj= 0; indj<N; indj++){
        if (indi!=indj){
            // Computing foce between particle i and j with neutons formula 
            jx = indj*2; jy = indj*2+1;
            radius = sqrt(pow(position[ix]-position[jx],2)+pow(position[iy]-position[jy],2));

            // Direction of force Particle 2 excerts on Particle1 (normalized)
            dir_x = (position[jx]-position[ix])/radius;
            dir_y = (position[jy]-position[iy])/radius;

            //Magnitude;
            forceMagnitude = G/(pow(radius,2)+10);//+1*0)-0/(pow(radius,4)+0.5));

            force_x += dir_x*forceMagnitude;
            force_y += dir_y*forceMagnitude;

        }
    }
    //Star At Origin Weighing 99% of the systems mass
    radius = sqrt(pow(position[ix],2)+pow(position[iy],2));
    dir_x = -position[ix]/radius;
    dir_y = -position[iy]/radius;
    forceMagnitude = G*N*99/(pow(radius,2));

    force_x += dir_x*forceMagnitude;
    force_y += dir_y*forceMagnitude;

    velocity[ix] += dt*force_x;
    velocity[iy] += dt*force_y;
    position[ix] += dt*velocity[ix];
    position[iy] += dt*velocity[iy]; 
}

kernel void getIndices(const float zoom,
                global float *position,
                global uchar *image,
                global uint *index,
                const uint N,
                const uint F,
                const uint G)
{
    uint ind = get_global_id(0);
    index[ind*2  ] = (int) position[ind*2  ]*zoom+F/2;
    index[ind*2+1] = (int) position[ind*2+1]*zoom+G/2;
}

kernel void computeImage(global float *position,
                global uchar *image,
                global uint *index,
                const uint N,
                const uint F,
                const uint G)
{   
    uint f = get_global_id(0);
    uint g = get_global_id(1);

    image[f*G+g] = 0;
    for(uint ind= 0; ind<N; ind++){
        if(index[ind*2  ]==f){
            if(index[ind*2+1]==g){
                image[f*G+g] +=1;
            }
        } 
    }
}

