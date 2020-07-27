#include "../launchParamsGlobal.h"
#include  "../util.h"

struct GlobalParams{
    float4 lightPos;
    float4 lightDir;
    float4 *accumBuffer;
    int shadowRays;
    float gamma;
    float lightScale;
    float glossiness;
    bool russian_roulette;
} ;


struct LaunchParams
{
    Frame frame;
    Camera camera;
    OptixTraversableHandle traversable;

    GlobalParams *global;
};

