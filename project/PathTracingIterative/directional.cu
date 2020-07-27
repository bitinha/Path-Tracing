#include "optixParams.h" // our launch params


extern "C" {
    __constant__ LaunchParams optixLaunchParams;
}

// ray types
enum { RAIDANCE = 0, SHADOW, RAY_TYPE_COUNT };

struct RadiancePRD {
    float3   emitted;
    float3   radiance;
    float3   attenuation;
    float3   origin;
    float3   direction;
    bool done;
    uint32_t seed;
    int32_t  countEmitted;
};

struct shadowPRD {
    float3 shadowAtt;
    uint32_t seed;
    int depth;
};




// -------------------------------------------------------

extern "C" __global__ void __closesthit__radiance() {

    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));
    // intersection position
    const float3& rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);


    uint32_t seed = prd.seed;



    float3 diffuseColor;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
            = (1.f - u - v) * sbtData.vertexD.texCoord0[index.x]
            + u * sbtData.vertexD.texCoord0[index.y]
            + v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor = make_float3(fromTexture);

    }
    else
        diffuseColor = sbtData.diffuse;




    const float r = rnd(seed);

    float continue_probability;
    if (optixLaunchParams.global->russian_roulette) {
        continue_probability = (diffuseColor.x + diffuseColor.y + diffuseColor.z) / 3;
    }
    else
    {
        continue_probability = 1.0f;
    }

    if (continue_probability <= r) {
        prd.done = true;
        return;
    }


    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;


    // Assumir que a luz vem diretamente de cima
    float3 lightDir = make_float3(normalize(optixLaunchParams.global->lightDir));
    
        
    shadowPRD shadow_prd;
    shadow_prd.shadowAtt = make_float3(0);
    shadow_prd.depth = 0;
    {

        uint32_t u0, u1;
        packPointer(&shadow_prd, u0, u1);

        optixTrace(optixLaunchParams.traversable,
            pos,
            -lightDir,
            0.001f,         // tmin
            1e20f,          // tmax
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SHADOW,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            SHADOW,      // missSBTIndex
            u0, u1);

    }

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(nn);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = pos;

        float spec_frac = pow(clamp(dot(normalize(optixGetWorldRayDirection()), normalize(reflect(-lightDir, nn))), 0.0, 1.0), sbtData.shininess);
        float3 brdf_spec = /*(sbtData.shininess + 2) / 2 **/ sbtData.specular * spec_frac;
        //Impedir que seja superior a 1 e verificar se está na sombra
        brdf_spec *= shadow_prd.shadowAtt;
        float3 brdf = diffuseColor/**(1-spec_frac)*/ + brdf_spec;
        brdf = make_float3(min(1.0, brdf.x), min(1.0, brdf.y), min(1.0, brdf.z));
        prd.attenuation *= brdf;
        prd.countEmitted = false;
    }

    prd.radiance += shadow_prd.shadowAtt * optixLaunchParams.global->lightScale / continue_probability;
}


extern "C" __global__ void __anyhit__radiance() {

}


// miss sets the background color
extern "C" __global__ void __miss__radiance() {

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    if (prd.countEmitted){
        prd.radiance = make_float3(0.529, 0.808, 0.922);
    }else{
        prd.radiance = make_float3(0);
    }
    prd.done = true;
}


// -----------------------------------------------
// Shadow rays

extern "C" __global__ void __closesthit__shadow() {

    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = make_float3(0.0f);
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow() {

    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = make_float3(1.0f);
}






// -----------------------------------------------
// Metal Phong rays

extern "C" __global__ void __closesthit__phong_metal() {


    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));
    // intersection position
    const float3& rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);

    uint32_t seed = prd.seed;
    const float r = rnd(seed);
    if ((sbtData.specular.x + sbtData.specular.y + sbtData.specular.z) / 3 < r) {
        prd.done = true;
        return;
    }


    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;


    const float glossiness = optixLaunchParams.global->glossiness;

    float3 reflectDir = reflect(optixGetWorldRayDirection(), nn);

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_power_sample_hemisphere(z1, z2, w_in, glossiness);
        Onb onb(reflectDir);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = pos;


        prd.attenuation *= sbtData.specular;
        prd.countEmitted = false;
    }

}







// -----------------------------------------------
// Light material


extern "C" __global__ void __closesthit__light() {


    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);

    prd.countEmitted = false;
    prd.radiance = sbtData.diffuse * optixLaunchParams.global->lightScale;
}


extern "C" __global__ void __anyhit__light() {
}


extern "C" __global__ void __miss__light() {
    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd.done = true;
}


extern "C" __global__ void __closesthit__light_shadow() {

    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    float3 intersectionPoint = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    float ndotl = max(0.0f, dot(normalize(make_float3(n)), -normalize(intersectionPoint - optixGetWorldRayOrigin())));
    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = ndotl * sbtData.emission;
}



extern "C" __global__ void __anyhit__light_shadow() {
}


extern "C" __global__ void __miss__light_shadow() {
    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = make_float3(0.0f);
}









// -----------------------------------------------
// Glass Phong rays


extern "C" __global__ void __closesthit__phong_glass() {


    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));
    // intersection position
    const float3& rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);


    uint32_t seed = prd.seed;



    float3 diffuseColor;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
            = (1.f - u - v) * sbtData.vertexD.texCoord0[index.x]
            + u * sbtData.vertexD.texCoord0[index.y]
            + v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor = make_float3(fromTexture);

    }
    else
        diffuseColor = sbtData.diffuse;


    const float r = rnd(seed);
    float continue_probability;
    if (optixLaunchParams.global->russian_roulette) {
        continue_probability = (diffuseColor.x + diffuseColor.y + diffuseColor.z) / 3;
    }
    else
    {
        continue_probability = 1.0f;
    }

    if (continue_probability < r) {
        prd.done = true;
        return;
    }


    {
        const float z1 = rnd(seed);
        prd.seed = seed;

        // new ray direction
        float3 rayDir = make_float3(0);

        const float3 normRayDir = optixGetWorldRayDirection();

        float indice_refracao_1;
        float indice_refracao_2;
        float3 r_normal;

        if (dot(normRayDir, nn) < 0) {
            indice_refracao_1 = 1;
            indice_refracao_2 = 1.5;
            r_normal = nn;
        }
        else {
            indice_refracao_1 = 1.5;
            indice_refracao_2 = 1;
            r_normal = -nn;
        }

        float costeta_i = dot(normalize(normRayDir), normalize(-r_normal));
        float costeta_t = sqrt(1 - (indice_refracao_1 / indice_refracao_2) * (indice_refracao_1 / indice_refracao_2) * (1 - (costeta_i) * (costeta_i)));

        float rs = (indice_refracao_2 * costeta_i - indice_refracao_1 * costeta_t) / (indice_refracao_2 * costeta_i + indice_refracao_1 * costeta_t) * (indice_refracao_2 * costeta_i - indice_refracao_1 * costeta_t) / (indice_refracao_2 * costeta_i + indice_refracao_1 * costeta_t);
        float rp = ((indice_refracao_2 * costeta_t - indice_refracao_1 * costeta_i) / (indice_refracao_2 * costeta_t + indice_refracao_1 * costeta_i)) * (indice_refracao_2 * costeta_t - indice_refracao_1 * costeta_i) / (indice_refracao_2 * costeta_t + indice_refracao_1 * costeta_i);
        float fr = (rs + rp) / 2.0;

        // Determinar se deve refratar
        if (z1 > fr) {
            rayDir = refract(normRayDir, r_normal, 1); //Vidro é apenas definido por um plano, por isso é apenas feita a transmissao raio sem desvios
        }
        // Caso a refração não seja possível
        if (length(rayDir) == 0) {
            rayDir = reflect(normRayDir, r_normal);
        }

        prd.direction = rayDir;
        prd.origin = pos;

        prd.attenuation *= diffuseColor;
        //prd.countEmitted = false;
    }


    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;



}



extern "C" __global__ void __anyhit__phong_glass() {

}


// miss sets the background color
extern "C" __global__ void __miss__phong_glass() {

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();
    prd.radiance = make_float3(0.0f, 0.0f, 0.0f);
    prd.done = true;
}



// -----------------------------------------------
// Glass Shadow rays

extern "C" __global__ void __closesthit__shadow_glass() {

    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();

    if (prd.depth > optixLaunchParams.global->shadowRays) {
        prd.shadowAtt = make_float3(0);
        return;
    }
    // ray payload
    shadowPRD afterPRD;
    afterPRD.shadowAtt = make_float3(1.0f);
    afterPRD.depth = prd.depth + 1;
    afterPRD.seed = prd.seed;
    uint32_t u0, u1;
    packPointer(&afterPRD, u0, u1);

    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();


    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));

    float indice_refracao_1;
    float indice_refracao_2;
    float3 r_normal;
    float3 normRayDir = optixGetWorldRayDirection();

    if (dot(normRayDir, nn) < 0) {
        indice_refracao_1 = 1;
        indice_refracao_2 = 1;
        r_normal = nn;

    }
    else {
        indice_refracao_1 = 1;
        indice_refracao_2 = 1;
        r_normal = -nn;
    }

    float3 rayDir = make_float3(0);

    uint32_t seed = prd.seed;

    const float z1 = rnd(seed);
    prd.seed = seed;

    float costeta_i = dot(normalize(normRayDir), normalize(-r_normal));
    float costeta_t = sqrt(1 - (indice_refracao_1 / indice_refracao_2) * (indice_refracao_1 / indice_refracao_2) * (1 - (costeta_i) * (costeta_i)));

    float rs = (indice_refracao_2 * costeta_i - indice_refracao_1 * costeta_t) / (indice_refracao_2 * costeta_i + indice_refracao_1 * costeta_t) * (indice_refracao_2 * costeta_i - indice_refracao_1 * costeta_t) / (indice_refracao_2 * costeta_i + indice_refracao_1 * costeta_t);
    float rp = ((indice_refracao_2 * costeta_t - indice_refracao_1 * costeta_i) / (indice_refracao_2 * costeta_t + indice_refracao_1 * costeta_i)) * (indice_refracao_2 * costeta_t - indice_refracao_1 * costeta_i) / (indice_refracao_2 * costeta_t + indice_refracao_1 * costeta_i);
    float fr = (rs + rp) / 2.0;

    // Determinar se deve refratar
    if (z1 > fr) {
        rayDir = refract(normRayDir, r_normal, indice_refracao_1 / indice_refracao_2);
    }
    // Caso a refração não seja possível
    if (length(rayDir) == 0) {
        rayDir = reflect(normRayDir, r_normal);
    }

    // trace primary ray
    optixTrace(optixLaunchParams.traversable,
        pos,
        rayDir,
        0.001f,    // tmin
        1e20f,  // tmax
        0.0f,   // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT, //OPTIX_RAY_FLAG_NONE,
        SHADOW,             // SBT offset
        RAY_TYPE_COUNT,     // SBT stride
        SHADOW,             // missSBTIndex 
        u0, u1);

    prd.shadowAtt = /*0.95f*/sbtData.diffuse * afterPRD.shadowAtt;
}


// any hit for shadows
extern "C" __global__ void __anyhit__shadow_glass() {

}


// miss for shadows
extern "C" __global__ void __miss__shadow_glass() {

    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();
    prd.shadowAtt = make_float3(0.0f);
}










extern "C" __global__ void __closesthit__phong_alphaTrans()
{

    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    RadiancePRD& prd = *(RadiancePRD*)getPRD<RadiancePRD>();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // compute normal
    const float4 n
        = (1.f - u - v) * sbtData.vertexD.normal[index.x]
        + u * sbtData.vertexD.normal[index.y]
        + v * sbtData.vertexD.normal[index.z];

    const float3 nn = normalize(make_float3(n));
    // intersection position
    const float3& rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;


    if (prd.countEmitted && length(sbtData.emission) != 0) {
        prd.emitted = sbtData.emission;
        return;
    }
    else
        prd.emitted = make_float3(0.0f);


    uint32_t seed = prd.seed;



    float4 diffuseColor;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
            = (1.f - u - v) * sbtData.vertexD.texCoord0[index.x]
            + u * sbtData.vertexD.texCoord0[index.y]
            + v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor = fromTexture;

    }
    else
        diffuseColor = make_float4(sbtData.diffuse, 1);




    const float r = rnd(seed);

    float continue_probability;
    if (optixLaunchParams.global->russian_roulette) {
        continue_probability = (diffuseColor.x + diffuseColor.y + diffuseColor.z) / 3;
    }
    else
    {
        continue_probability = 1.0f;
    }

    if (continue_probability <= r) {
        prd.done = true;
        return;
    }


    const float z1 = rnd(seed);
    const float z2 = rnd(seed);
    prd.seed = seed;


    // Assumir que a luz vem diretamente de cima
    float3 lightDir = make_float3(normalize(optixLaunchParams.global->lightDir));


    shadowPRD shadow_prd;
    shadow_prd.shadowAtt = make_float3(0);
    shadow_prd.depth = 0;
    if (diffuseColor.w > 0.5) {

        uint32_t u0, u1;
        packPointer(&shadow_prd, u0, u1);

        optixTrace(optixLaunchParams.traversable,
            pos,
            -lightDir,
            0.001f,         // tmin
            1e20f,          // tmax
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SHADOW,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            SHADOW,      // missSBTIndex
            u0, u1);

        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        float3 w_in;
        cosine_sample_hemisphere(z1, z2, w_in);
        Onb onb(nn);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = pos;

        float spec_frac = pow(clamp(dot(normalize(optixGetWorldRayDirection()), normalize(reflect(-lightDir, nn))), 0.0, 1.0), sbtData.shininess);
        float3 brdf_spec = /*(sbtData.shininess + 2) / 2 **/ sbtData.specular * spec_frac;
        //Impedir que seja superior a 1 e verificar se está na sombra
        brdf_spec *= shadow_prd.shadowAtt;
        float3 brdf = make_float3(diffuseColor) /** (1 - spec_frac)*/ + brdf_spec;
        brdf = make_float3(min(1.0, brdf.x), min(1.0, brdf.y), min(1.0, brdf.z));
        prd.attenuation *= brdf;
        prd.countEmitted = false;
        prd.radiance += shadow_prd.shadowAtt * optixLaunchParams.global->lightScale / continue_probability;
    }
    else {

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(optixLaunchParams.traversable,
            pos,
            optixGetWorldRayDirection(),
            0.001f,    // tmin
            1e20f,  // tmax
            0.0f, OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE, RAIDANCE, RAY_TYPE_COUNT, RAIDANCE, u0, u1);
    }
}




extern "C" __global__ void __closesthit__shadow_alphaTrans() {

    const TriangleMeshSBTData& sbtData
        = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();

    // retrieve primitive id and indexes
    const int   primID = optixGetPrimitiveIndex();
    const uint3 index = sbtData.index[primID];

    // get barycentric coordinates
    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    // intersection position
    const float3& rayDir = optixGetWorldRayDirection();
    const float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;

    float4 diffuseColor;

    if (sbtData.hasTexture && sbtData.vertexD.texCoord0) {

        // compute pixel texture coordinate
        const float4 tc
            = (1.f - u - v) * sbtData.vertexD.texCoord0[index.x]
            + u * sbtData.vertexD.texCoord0[index.y]
            + v * sbtData.vertexD.texCoord0[index.z];
        // fetch texture value
        float4 fromTexture = tex2D<float4>(sbtData.texture, tc.x, tc.y);
        diffuseColor = fromTexture;

    }
    else
        diffuseColor = make_float4(sbtData.diffuse, 1);



    shadowPRD& prd = *(shadowPRD*)getPRD<shadowPRD>();

    if (diffuseColor.w > 0.5) {
        prd.shadowAtt = make_float3(0.0f);
    }
    else {

        uint32_t u0, u1;
        packPointer(&prd, u0, u1);

        optixTrace(optixLaunchParams.traversable,
            pos,
            rayDir,
            0.001f,         // tmin
            1e20f,          // tmax
            0.0f,                    // rayTime
            OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            SHADOW,      // SBT offset
            RAY_TYPE_COUNT,          // SBT stride
            SHADOW,      // missSBTIndex
            u0, u1);

    }

}






// -----------------------------------------------
// Primary Rays


extern "C" __global__ void __raygen__renderFrame() {

    const int ix = optixGetLaunchIndex().x;
    const int iy = optixGetLaunchIndex().y;
    const auto& camera = optixLaunchParams.camera;

    const int& maxDepth = optixLaunchParams.frame.maxDepth;

    float squaredRaysPerPixel = float(optixLaunchParams.frame.raysPerPixel);
    float2 delta = make_float2(1.0f / squaredRaysPerPixel, 1.0f / squaredRaysPerPixel);

    float3 result = make_float3(0.0f);

    uint32_t seed = tea<4>(ix * optixGetLaunchDimensions().x + iy, optixLaunchParams.frame.frame);

    for (int i = 0; i < squaredRaysPerPixel; ++i) {
        for (int j = 0; j < squaredRaysPerPixel; ++j) {

            const float2 subpixel_jitter = make_float2(delta.x * (i + rnd(seed)), delta.y * (j + rnd(seed)));
            const float2 screen(make_float2(ix + subpixel_jitter.x, iy + subpixel_jitter.y)
                / make_float2(optixGetLaunchDimensions().x, optixGetLaunchDimensions().y) * 2.0 - 1.0);

            // note: nau already takes into account the field of view and ratio when computing 
            // camera horizontal and vertical
            float3 origin = camera.position;
            float3 rayDir = normalize(camera.direction
                + (screen.x) * camera.horizontal
                + (screen.y) * camera.vertical);

            RadiancePRD prd;
            prd.emitted = make_float3(0.f);
            prd.radiance = make_float3(0.f);
            prd.attenuation = make_float3(1.f);
            prd.countEmitted = true;
            prd.done = false;
            prd.seed = seed;

            uint32_t u0, u1;
            packPointer(&prd, u0, u1);

            for (int k = 0; k < maxDepth && !prd.done; ++k) {

                optixTrace(optixLaunchParams.traversable,
                    origin,
                    rayDir,
                    0.001f,    // tmin
                    1e20f,  // tmax
                    0.0f, OptixVisibilityMask(1),
                    OPTIX_RAY_FLAG_DISABLE_ANYHIT, RAIDANCE, RAY_TYPE_COUNT, RAIDANCE, u0, u1);

                result += prd.emitted;
                result += prd.radiance * prd.attenuation;

                origin = prd.origin;
                rayDir = prd.direction;

            }
        }
    }

    result = result / (squaredRaysPerPixel * squaredRaysPerPixel);
    float gamma = optixLaunchParams.global->gamma;
    // compute index
    const uint32_t fbIndex = ix + iy * optixGetLaunchDimensions().x;

    optixLaunchParams.global->accumBuffer[fbIndex] =
        (optixLaunchParams.global->accumBuffer[fbIndex] * optixLaunchParams.frame.subFrame +
            make_float4(result.x, result.y, result.z, 1)) / (optixLaunchParams.frame.subFrame + 1);


    float4 rgbaf = optixLaunchParams.global->accumBuffer[fbIndex];
    //convert float (0-1) to int (0-255)
    const int r = int(255.0f * min(1.0f, pow(rgbaf.x, 1 / gamma)));
    const int g = int(255.0f * min(1.0f, pow(rgbaf.y, 1 / gamma)));
    const int b = int(255.0f * min(1.0f, pow(rgbaf.z, 1 / gamma)));

    // convert to 32-bit rgba value 
    const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
    // write to output buffer
    optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
}



