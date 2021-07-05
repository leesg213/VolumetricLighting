/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Metal shaders used for this sample.
*/

#include <metal_stdlib>

using namespace metal;

#include "AAPLShaderTypes.h"

struct Vertex
{
    float3 position  [[attribute(VertexAttributePosition)]];
    float2 texCoord  [[attribute(VertexAttributeTexcoord)]];
    half3  normal    [[attribute(VertexAttributeNormal)]];
    half3  tangent   [[attribute(VertexAttributeTangent)]];
    half3  bitangent [[attribute(VertexAttributeBitangent)]];
};

struct ColorInOut
{
    float4 position [[position]];
    float4 currentFramePosition;
    float4 prevFramePosition;
    float2 texCoord;

    half3  worldPos;
    half3  tangent;
    half3  bitangent;
    half3  normal;
};

struct FragmentOut
{
    float4 color [[color(0)]];
    float4 velocity [[color(1)]];
};

// Vertex function
vertex ColorInOut vertexTransform (const Vertex in                               [[ stage_in ]],
                                   const uint   instanceId                       [[ instance_id ]],
                                   const device ActorParams&    actorParams      [[ buffer (BufferIndexActorParams)    ]],
                                   constant     ViewportParams& viewportParams   [[ buffer (BufferIndexViewportParams) ]] )
{
    ColorInOut out;
    out.texCoord = in.texCoord;

    float4x4 currentFrame_modelMatrix = actorParams.modelMatrix;
    float4 currentFrame_worldPos  = currentFrame_modelMatrix * float4(in.position, 1.0);
    float4 currentFrame_clipPos = viewportParams.viewProjectionMatrix * currentFrame_worldPos;
    
    out.worldPos = half3(currentFrame_worldPos.xyz);
    out.currentFramePosition = currentFrame_clipPos;
    
    float4 currentFrame_clipPos_jittered =
        currentFrame_clipPos + float4(viewportParams.jitter*currentFrame_clipPos.w,0,0);
    
    out.position = currentFrame_clipPos_jittered;
    
    float4x4 prevFrame_modelMatrix = actorParams.prevModelMatrix;
    float4 prevFrame_worldPos  = prevFrame_modelMatrix * float4(in.position, 1.0);
    float4 prevFrame_clipPos = viewportParams.prevViewProjMatrix * prevFrame_worldPos;
    
    out.prevFramePosition = prevFrame_clipPos;

    half3x3 normalMatrix = half3x3((half3)currentFrame_modelMatrix[0].xyz,
                                   (half3)currentFrame_modelMatrix[1].xyz,
                                   (half3)currentFrame_modelMatrix[2].xyz);

    out.tangent   = normalMatrix * in.tangent;
    out.bitangent = normalMatrix * in.bitangent;
    out.normal    = normalMatrix * in.normal;

    return out;
}

float2 CalcVelocity(float4 newPos, float4 oldPos, float2 viewSize)
{
    oldPos /= oldPos.w;
    oldPos.xy = (oldPos.xy+1)/2.0f;
    oldPos.y = 1 - oldPos.y;
    
    newPos /= newPos.w;
    newPos.xy = (newPos.xy+1)/2.0f;
    newPos.y = 1 - newPos.y;
    
    return (newPos - oldPos).xy;
}

float CalcShadow(float3 worldPos,
                 constant FrameParams& frameParams,
                 texture2d<float> shadowMap)
{
    float4 pos_in_lightspace = frameParams.shadowMapViewProjMatrix * float4(worldPos,1);
    
    pos_in_lightspace /= pos_in_lightspace.w;
    
    float2 shadow_sample_pos = (pos_in_lightspace.xy +1)/2.0f;
    shadow_sample_pos.y = 1 - shadow_sample_pos.y;
    
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float shadowDepth = shadowMap.sample(sam, shadow_sample_pos).x+0.0025;
    
    return shadowDepth > pos_in_lightspace.z ? 1 : 0;
}
// Fragment function used to render the temple object in both the
//   reflection pass and the final pass
fragment FragmentOut fragmentLighting (         ColorInOut      in             [[ stage_in ]],
                                  device   ActorParams&    actorParams    [[ buffer (BufferIndexActorParams)    ]],
                                  constant FrameParams &   frameParams    [[ buffer (BufferIndexFrameParams)    ]],
                                  constant ViewportParams& viewportParams [[ buffer (BufferIndexViewportParams) ]],
                                           texture2d<half> baseColorMap   [[ texture (TextureIndexBaseColor)    ]],
                                           texture2d<half> normalMap      [[ texture (TextureIndexNormal)       ]],
                                           texture2d<half> specularMap    [[ texture (TextureIndexSpecular)     ]],
                                        texture2d<float> shadowMap [[texture(TextureIndexShadowMap)]]
                                       )
{
    constexpr sampler linearSampler (mip_filter::linear,
                                     mag_filter::linear,
                                     min_filter::linear);
    FragmentOut out;

    const half4 baseColorSample = baseColorMap.sample (linearSampler, in.texCoord.xy);
    half3 normalSampleRaw = normalMap.sample (linearSampler, in.texCoord.xy).xyz;
    // The x and y coordinates in a normal map (red and green channels) are mapped from [-1;1] to [0;255].
    // As the sampler returns a value in [0 ; 1], we need to do :
    normalSampleRaw.xy = normalSampleRaw.xy * 2.0 - 1.0;
    const half3 normalSample = normalize(normalSampleRaw);

    const half  specularSample  = specularMap.sample  (linearSampler, in.texCoord.xy).x*0.5;

    // The per-vertex vectors have been interpolated, thus we need to normalize them again :
    in.tangent   = normalize (in.tangent);
    in.bitangent = normalize (in.bitangent);
    in.normal    = normalize (in.normal);

    half3x3 tangentMatrix = half3x3(in.tangent, in.bitangent, in.normal);

    float3 normal = (float3) (tangentMatrix * normalSample);

    float shadowTerm = CalcShadow(float3(in.worldPos), frameParams, shadowMap);
    
    float3 directionalContribution = float3(0);
    float3 specularTerm = float3(0);
    {
        float nDotL = saturate (dot(normal, frameParams.directionalLightInvDirection));

        // The diffuse term is the product of the light color, the surface material
        // reflectance, and the falloff
        float3 diffuseTerm = frameParams.directionalLightColor * nDotL * shadowTerm;

        // Apply specular lighting...

        // 1) Calculate the halfway vector between the light direction and the direction they eye is looking
        float3 eyeDir = normalize (viewportParams.cameraPos - float3(in.worldPos));
        float3 halfwayVector = normalize(frameParams.directionalLightInvDirection + eyeDir);

        // 2) Calculate the reflection amount by evaluating how the halfway vector matches the surface normal
        float reflectionAmount = saturate(dot(normal, halfwayVector));

        // 3) Calculate the specular intensity by powering our reflection amount to our object's
        //    shininess
        float specularIntensity = powr(reflectionAmount*1.025, actorParams.materialShininess*4);

        // 4) Obtain the specular term by multiplying the intensity by our light's color
        specularTerm = frameParams.directionalLightColor * specularIntensity * float(specularSample) * shadowTerm;

        // The base color sample is actually the diffuse color of the material
        float3 baseColor = float3(baseColorSample.xyz) * actorParams.diffuseMultiplier;

        // The ambient contribution is an approximation for global, indirect lighting, and simply added
        //   to the calculated lit color value below

        // Calculate diffuse contribution from this light : the sum of the diffuse and ambient * albedo
        directionalContribution = baseColor * (diffuseTerm + frameParams.ambientLightColor);
    }

    // Now that we have the contributions our light sources in the scene, we sum them together
    //   to get the fragment's lit color value
    float3 color = specularTerm + directionalContribution;

    // We return the color we just computed and the alpha channel of our baseColorMap for this
    //   fragment's alpha value
    
    out.color = float4(color, baseColorSample.w);
    out.velocity = float4(CalcVelocity(in.currentFramePosition, in.prevFramePosition, viewportParams.viewSize),0,0);
    
    return out;
}

fragment FragmentOut fragmentGround (         ColorInOut      in             [[ stage_in ]],
                                     constant FrameParams &   frameParams    [[ buffer (BufferIndexFrameParams)    ]],
                                constant ViewportParams& viewportParams [[ buffer (BufferIndexViewportParams) ]],
                                     texture2d<float> shadowMap [[texture(TextureIndexShadowMap)]] )
{
    float onEdge;
    {
        float2 onEdge2d = fract(float2(in.worldPos.xz)/500.f);
        // If onEdge2d is negative, we want 1. Otherwise, we want zero (independent for each axis).
        float2 offset2d = sign(onEdge2d) * -0.5 + 0.5;
        onEdge2d += offset2d;
        onEdge2d = step (0.03, onEdge2d);

        onEdge = min(onEdge2d.x, onEdge2d.y);
    }
    
    float shadowTerm = CalcShadow(float3(in.worldPos), frameParams, shadowMap)*0.5f+0.5f;

    float3 neutralColor = float3 (1, 1, 0)*0.15;
    float3 edgeColor = neutralColor * 0.2;
    float3 groundColor = mix (edgeColor, neutralColor, 1) * shadowTerm;

    FragmentOut out;
    
    out.color = float4(groundColor, 1);

    out.velocity = float4(CalcVelocity(in.currentFramePosition, in.prevFramePosition, viewportParams.viewSize),0,0);
    
    return out;
}


// Screen filling quad in normalized device coordinates
constant float2 quadVertices[] = {
    float2(-1, -1),
    float2(-1,  1),
    float2( 1,  1),
    float2(-1, -1),
    float2( 1,  1),
    float2( 1, -1)
};

struct quadVertexOut {
    float4 position [[position]];
    float2 uv;
};

// Simple vertex shader which passes through NDC quad positions
vertex quadVertexOut fullscreenQuadVertex(unsigned short vid [[vertex_id]]) {
    float2 position = quadVertices[vid];
    
    quadVertexOut out;
    
    out.position = float4(position, 0, 1);
    out.uv = position * 0.5f + 0.5f;
    out.uv.y = 1 - out.uv.y;
    return out;
}

float2 FindClosestDepthSamplePos(float2 uv, texture2d<float> depthbuffer, float2 viewSize)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float2 closest = uv;
    float closest_depth = 1;
    
    float depth = 0;
    
    for(int y = -1;y<=1;++y)
    {
        for(int x=-1;x<=1;++x)
        {
            float2 uv_offset = float2(x,y) / viewSize;
            depth = depthbuffer.sample(sam, uv + uv_offset).x;
            if(depth < closest_depth)
            {
                closest_depth = depth;
                closest = uv + uv_offset;
            }
        }
    }
    
    return closest;
}
// Simple fragment shader which copies a texture and applies a simple tonemapping function
fragment float4 TAA_ResolveFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> currentFrameColorBuffer [[texture(0)]],
                             texture2d<float> historyBuffer [[texture(1)]],
                            texture2d<float> velocityBuffer [[texture(2)]])
{
    constexpr sampler sam_point(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    constexpr sampler sam_linear(min_filter::linear, mag_filter::linear, mip_filter::none);

    float2 velocity_sample_pos = in.uv;
    float2 velocity = velocityBuffer.sample(sam_point, velocity_sample_pos).xy;
    float2 prevousPixelPos = in.uv - velocity;
    
    float3 currentColor = currentFrameColorBuffer.sample(sam_point, in.uv).xyz;
    float3 historyColor = historyBuffer.sample(sam_linear, prevousPixelPos).xyz;

    // Apply clamping on the history color.
    float3 NearColor0 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(1, 0)).xyz;
    float3 NearColor1 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(0, 1)).xyz;
    float3 NearColor2 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(-1, 0)).xyz;
    float3 NearColor3 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(0, -1)).xyz;
    float3 NearColor4 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(1, 1)).xyz;
    float3 NearColor5 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(-1, 1)).xyz;
    float3 NearColor6 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(1, -1)).xyz;
    float3 NearColor7 = currentFrameColorBuffer.sample(sam_point, in.uv, int2(-1, -1)).xyz;

    float3 BoxMin = min(currentColor, min(NearColor0, min(NearColor1, min(NearColor2, NearColor3))));
    float3 BoxMax = max(currentColor, max(NearColor0, max(NearColor1, max(NearColor2, NearColor3))));;

    BoxMin = min(BoxMin, min(NearColor4, min(NearColor5, min(NearColor6, NearColor7))));
    BoxMax = max(BoxMax, max(NearColor4, max(NearColor5, max(NearColor6, NearColor7))));;

    historyColor = clamp(historyColor, BoxMin, BoxMax);
    
    float modulationFactor = 0.9;
    
    float3 color = mix(currentColor, historyColor, modulationFactor);

    return float4(color, 1.0f);
}

fragment float4 BlitFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    return float4(color, 1);
}

vertex quadVertexOut magnifierVertex(unsigned short vid [[vertex_id]],
                                     constant MagnifierParams& magnifierParam [[buffer(0)]]) {
    float2 position = quadVertices[vid];
    float2 uv = position;
    
    quadVertexOut out;
    
    float2 scale = magnifierParam.size / (magnifierParam.viewSize);
    
    float2 windowPos = magnifierParam.viewSize - magnifierParam.size/2;
    float2 translate = (windowPos / (magnifierParam.viewSize))*2 + float2(-1,-1);
    
    position.xy *= scale;
    position.xy += translate;
    
    uv *= scale;
    float2 uv_translate = (magnifierParam.position / magnifierParam.viewSize)*2 + float2(-1,-1);
    uv += uv_translate;
    uv = uv*0.5f + 0.5f;
    float magnifier_scale = 6;
    float2 tex_center = uv_translate * 0.5f + 0.5f;
    uv = uv - tex_center;
    uv /= magnifier_scale;
    uv = uv + tex_center;
    uv.y = 1 - uv.y;
    out.uv = uv;
    out.position = float4(position, 0, 1);
    return out;
}

fragment float4 magnifierFragment(quadVertexOut in [[stage_in]],
                             texture2d<float> tex)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float3 color = tex.sample(sam, in.uv).xyz;
    
    return float4(color, 1);
}

struct ShadowMapVertexOut
{
    float4 position [[position]];
};


vertex ShadowMapVertexOut shadowMapVertex (const Vertex in                               [[ stage_in ]],
                                   const uint   instanceId                       [[ instance_id ]],
                                   const device ActorParams&    actorParams      [[ buffer (BufferIndexActorParams)    ]],
                                   constant     ViewportParams& viewportParams   [[ buffer (BufferIndexViewportParams) ]] )
{
    float4x4 currentFrame_modelMatrix = actorParams.modelMatrix;
    float4 currentFrame_worldPos  = currentFrame_modelMatrix * float4(in.position, 1.0);
    float4 currentFrame_clipPos = viewportParams.viewProjectionMatrix * currentFrame_worldPos;
    
    ShadowMapVertexOut out;
    out.position = currentFrame_clipPos;
    return out;
}


fragment float4 shadowMapFragment()
{
    return float4(1,1,1,1);
}


float calcScatteringFactor(float3 lightDir, float3 pos, float3 cameraPos)
{
    float3 vToEye = normalize(cameraPos - pos);
    float cos_theta = max(0.0, dot(vToEye, lightDir));
    
    float g = 0.1f;
    float f_hg = pow(1-g,2) / sqrt(pow(4*3.141592f*(1 + g*g - 2*g*cos_theta),3));
    
    return f_hg * 50;
}

float volumeLightingTraceRay(float3 start,
                             float3 end,
                             constant VolumeLightingParams &   volumeLightingParams,
                             texture2d<float> shadowMapBuffer,
                             float ditherOffset)
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    const float stepDistane = volumeLightingParams.stepSize;
    
    float remainDistance = distance(start, end);
    float3 direction = (end-start)/remainDistance;
    
    float3 pos = start + direction*stepDistane * ditherOffset;
    float intensity = 0;
    int numSteps = 0;
    int numShadows = 0;
    
    float3 sceneCenter = float3(0.f, -250.f, 1000.f);
    
    while(remainDistance>0)
    {
        float curStepDistance = min(remainDistance, stepDistane);
        
        pos += direction * curStepDistance;
        remainDistance -= curStepDistance;
        
        float4 posInShadowClipSpace = volumeLightingParams.shadowMapViewProjMatrix * float4(pos, 1);
        posInShadowClipSpace /= posInShadowClipSpace.w;
        float2 shadowSamplePos = posInShadowClipSpace.xy;
        shadowSamplePos.y *= -1;
        shadowSamplePos = (shadowSamplePos+1)/2.0f;
      //  shadowSamplePos.y = 1 - shadowSamplePos.y;
        
        float depthInShadowSpace = shadowMapBuffer.sample(sam, shadowSamplePos).x;
        
        float shadow = posInShadowClipSpace.z > depthInShadowSpace ? 0 : 1;
        
        ++numSteps;
        numShadows += int(1 - shadow);
        
        float dist_from_scene_center = distance(sceneCenter, pos);
        shadow = dist_from_scene_center > 800 ? 0 : shadow;
        
        float scatterFactor = calcScatteringFactor(volumeLightingParams.lightDir,
                                                   pos,
                                                   volumeLightingParams.cameraPos);
        
        intensity += curStepDistance / 2000.0f * shadow * scatterFactor;
    }
    
    return pow(intensity,1);
}

fragment float4 volumeLightingFragment(quadVertexOut in [[stage_in]],
                                       constant VolumeLightingParams &   volumeLightingParams [[ buffer(0)]],
                                       texture2d<float> currentFrameColorBuffer [[texture(0)]],
                                       texture2d<float> depthBuffer [[texture(1)]],
                                      texture2d<float> shadowMapBuffer [[texture(2)]])
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    const int ditherOffsets[16] = {
        0,8,2,10,
        12,4,14,6,
        3,11,1,9,
        15,7,13,5
    };
    
    int div = 1;
    int ditherOffsetIndex = int(in.position.x/div)%4+(int(in.position.y/div)%4)*4 + volumeLightingParams.frameNumber;
    float ditherOffset = ditherOffsets[(ditherOffsetIndex)%16]/16.0;
    
    float3 frame_color = 0;//currentFrameColorBuffer.sample(sam, in.uv).xyz;
    float depth = depthBuffer.sample(sam, in.uv).x;
    float4 rayStart = float4(in.uv.x*2-1, (1-in.uv.y)*2-1, 0, 1);
    float4 rayEnd =float4(in.uv.x*2-1, (1-in.uv.y)*2-1, depth, 1);
    
   // rayStart.xy += volumeLightingParams.jitter;
   // rayEnd.xy += volumeLightingParams.jitter;
    
    rayStart = volumeLightingParams.invViewProjMatrix * rayStart;
    rayEnd = volumeLightingParams.invViewProjMatrix * rayEnd;
    
    float color = volumeLightingTraceRay(
                                         rayStart.xyz/rayStart.w,
                                         rayEnd.xyz/rayEnd.w,
                                         volumeLightingParams,
                                         shadowMapBuffer,
                                         ditherOffset);
    
    frame_color += color*float3(1,1,0);
    return float4(frame_color,1);
}

fragment float4 depthDownsample(quadVertexOut in [[stage_in]],
                                       texture2d<float> depthBuffer [[texture(0)]])
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float2 uv = (floor(in.position.xy) *2+0.5) / float2(1920, 1080);
    
    float4 depthSamples = depthBuffer.gather(sam, uv);
    
    float maxDepthSample = min(depthSamples.x, min(depthSamples.y, min(depthSamples.z, depthSamples.w)));
    
    return maxDepthSample;
}

fragment float4 composite(quadVertexOut in [[stage_in]],
                                       texture2d<float> frameColorTex [[texture(0)]],
                          texture2d<float> volumeLightingTex [[texture(1)]],
                          texture2d<float> fullResDepth [[texture(2)]],
                          texture2d<float> halfResDepth [[texture(3)]])
{
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    constexpr sampler samLinear(min_filter::linear, mag_filter::linear, mip_filter::none);

    float depth = fullResDepth.sample(sam, in.uv).x;
    float half_Depth = halfResDepth.sample(sam, in.uv).x;
    
    float half_depth_00 = halfResDepth.sample(sam, in.uv, int2(-1,0)).x;
    float half_depth_10 = halfResDepth.sample(sam, in.uv, int2(1,0)).x;
    float half_depth_01 = halfResDepth.sample(sam, in.uv, int2(0,1)).x;
    float half_depth_11 = halfResDepth.sample(sam, in.uv, int2(0,-1)).x;
    
    float diff = abs(half_Depth - depth);
    float diff_00 = abs(half_depth_00 - depth);
    float diff_10 = abs(half_depth_10 - depth);
    float diff_01 = abs(half_depth_01 - depth);
    float diff_11 = abs(half_depth_11 - depth);

    float min_diff = min(diff, min(diff_00, min(diff_01, min(diff_10, diff_11))));
    
    int2 volumeLightingSampleCoordOffset = 0;
    if(diff_00 == min_diff) volumeLightingSampleCoordOffset = int2(-1,0);
    if(diff_01 == min_diff) volumeLightingSampleCoordOffset = int2(0,1);
    if(diff_10 == min_diff) volumeLightingSampleCoordOffset = int2(1,0);
    if(diff_11 == min_diff) volumeLightingSampleCoordOffset = int2(0,-1);
    
    float3 frame = frameColorTex.sample(sam, in.uv).xyz;
    float3 volumeLighting = volumeLightingTex.sample(sam, in.uv, volumeLightingSampleCoordOffset).xyz;
    
    return float4(frame + volumeLighting, 1);
}

fragment float4 gaussian_blur(quadVertexOut in [[stage_in]],
                              constant GaussianBlurParams& blurParams [[buffer(0)]],
                                       texture2d<float> src [[texture(0)]],
                              texture2d<float> depthTex [[texture(1)]])
{
    const float weights[5] = { 0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216};
    constexpr sampler sam(min_filter::nearest, mag_filter::nearest, mip_filter::none);
    
    float4 result = src.sample(sam, in.uv) * weights[0];
    float depth = depthTex.sample(sam, in.uv).x;
    
    float totalWeight = weights[0];
    float depth_threshold = 0.005;
    for(int i = 1;i<5;++i)
    {
        float depth0 = depthTex.sample(sam, in.uv, blurParams.direction*i).x;
        float diff = abs(depth0 - depth);
        float weight = diff < depth_threshold ? weights[i] : 0;
        result += src.sample(sam, in.uv, blurParams.direction*i) * weight;
        totalWeight += weight;
        
        float depth1 = depthTex.sample(sam, in.uv, -blurParams.direction*i).x;
        diff = abs(depth1 - depth);
        weight = diff < depth_threshold ? weights[i] : 0;
        result += src.sample(sam, in.uv, -blurParams.direction*i) * weight;
        totalWeight += weight;
    }
    
    return result * (1/totalWeight);
}
