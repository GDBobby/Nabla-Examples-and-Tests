#include "common.hlsl"
#include <nbl/builtin/hlsl/shapes/beziers.hlsl>
#include <nbl/builtin/hlsl/math/equations/quadratic.hlsl>
#include <nbl/builtin/hlsl/limits.hlsl>
#include <nbl/builtin/hlsl/algorithm.hlsl>
#include <nbl/builtin/hlsl/jit/device_capabilities.hlsl>
#include <nbl/builtin/hlsl/shapes/line.hlsl>
#include <nbl/builtin/hlsl/math/geometry.hlsl>
#include <nbl/builtin/hlsl/spirv_intrinsics/fragment_shader_pixel_interlock.hlsl>
#include <nbl/builtin/hlsl/text_rendering/msdf.hlsl>

// ------- Vertex Shader -------

// TODO[Lucas]: Move these functions to builtin hlsl functions (Even the shadertoy obb and aabb ones)
float cross2D(float2 a, float2 b)
{
    return determinant(float2x2(a,b));
}

float2 BezierTangent(float2 p0, float2 p1, float2 p2, float t)
{
    return 2.0 * (1.0 - t) * (p1 - p0) + 2.0 * t * (p2 - p1);
}

float2 QuadraticBezier(float2 p0, float2 p1, float2 p2, float t)
{
    return nbl::hlsl::shapes::QuadraticBezier<float>::construct(p0, p1, p2).evaluate(t);
}

ClipProjectionData getClipProjectionData(in MainObject mainObj)
{
    if (mainObj.clipProjectionAddress != InvalidClipProjectionAddress)
    {
        ClipProjectionData ret;
        ret.projectionToNDC = vk::RawBufferLoad<float64_t3x3>(mainObj.clipProjectionAddress, 8u);
        ret.minClipNDC      = vk::RawBufferLoad<float32_t2>(mainObj.clipProjectionAddress + sizeof(float64_t3x3), 8u);
        ret.maxClipNDC      = vk::RawBufferLoad<float32_t2>(mainObj.clipProjectionAddress + sizeof(float64_t3x3) + sizeof(float32_t2), 8u);
        return ret;
    }
    else
    {
        return globals.defaultClipProjection;
    }
}

double2 transformPointNdc(float64_t3x3 transformation, double2 point2d)
{
    return mul(transformation, float64_t3(point2d, 1)).xy;
}
double2 transformVectorNdc(float64_t3x3 transformation, double2 vector2d)
{
    return mul(transformation, float64_t3(vector2d, 0)).xy;
}
float2 transformPointScreenSpace(float64_t3x3 transformation, double2 point2d) 
{
    double2 ndc = transformPointNdc(transformation, point2d);
    return (float2)((ndc + 1.0) * 0.5 * globals.resolution);
}
float4 transformFromSreenSpaceToNdc(float2 pos)
{
    return float4((pos.xy / globals.resolution) * 2.0 - 1.0, 0.0f, 1.0f);
}

template<bool FragmentShaderPixelInterlock>
void dilateHatch(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV);

// Dilate with ease, our transparency algorithm will handle the overlaps easily with the help of FragmentShaderPixelInterlock
template<>
void dilateHatch<true>(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV)
{
    const float2 dilatationFactor = 1.0 + 2.0 * dilateRate;
    
    // cornerMultiplier stores the direction of the corner to dilate:
    // (-1,-1)|--|(1,-1)
    //        |  |
    // (-1,1) |--|(1,1)
    const float2 cornerMultiplier = float2(undilatedCorner * 2.0 - 1.0);
    outUV = float2((cornerMultiplier * dilatationFactor + 1.0) * 0.5);
    
    // vx/vy are vectors in direction of the box's axes and their length is equal to X pixels (X = globals.antiAliasingFactor + 1.0)
    // and we use them for dilation of X pixels in ndc space by adding them to the currentCorner in NDC space 
    const float2 vx = ndcAxisU * dilateRate.x;
    const float2 vy = ndcAxisV * dilateRate.y;
    outOffsetVec = vx * cornerMultiplier.x + vy * cornerMultiplier.y; // (0, 0) should do -vx-vy and (1, 1) should do +vx+vy
}

// Don't dilate which causes overlap of colors when no fragshaderInterlock which powers our transparency and overlap resolving algorithm
template<>
void dilateHatch<false>(out float2 outOffsetVec, out float2 outUV, const float2 undilatedCorner, const float2 dilateRate, const float2 ndcAxisU, const float2 ndcAxisV)
{
    outOffsetVec = float2(0.0f, 0.0f);
    outUV = undilatedCorner;
    // TODO: If it became a huge bummer on AMD devices we can consider dilating only in minor direction which may still avoid color overlaps
    // Or optionally we could dilate and stuff when we know this hatch is opaque (alpha = 1.0)
}

[shader("vertex")]
PSInput vs_main(uint vertexID : SV_VertexID)
{
    const uint vertexIdx = vertexID & 0x3u;
    const uint objectID = vertexID >> 2;

    DrawObject drawObj = drawObjects[objectID];

    ObjectType objType = (ObjectType)(drawObj.type_subsectionIdx & 0x0000FFFF);
    uint32_t subsectionIdx = drawObj.type_subsectionIdx >> 16;
    PSInput outV;

    // Default Initialize PS Input
    outV.position.z = 0.0;
    outV.data1 = uint4(0, 0, 0, 0);
    outV.data2 = float4(0, 0, 0, 0);
    outV.data3 = float4(0, 0, 0, 0);
    outV.data4 = float4(0, 0, 0, 0);
    outV.interp_data5 = float2(0, 0);
    outV.setObjType(objType);
    outV.setMainObjectIdx(drawObj.mainObjIndex);
    
    MainObject mainObj = mainObjects[drawObj.mainObjIndex];
    ClipProjectionData clipProjectionData = getClipProjectionData(mainObj);
    
    // We only need these for Outline type objects like lines and bezier curves
    if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
    {
        LineStyle lineStyle = lineStyles[mainObj.styleIdx];

        // Width is on both sides, thickness is one one side of the curve (div by 2.0f)
        const float screenSpaceLineWidth = lineStyle.screenSpaceLineWidth + float(lineStyle.worldSpaceLineWidth * globals.screenToWorldRatio);
        const float antiAliasedLineThickness = screenSpaceLineWidth * 0.5f + globals.antiAliasingFactor;
        const float sdfLineThickness = screenSpaceLineWidth / 2.0f;
        outV.setLineThickness(sdfLineThickness);
        outV.setCurrentWorldToScreenRatio((float)(2.0 / (clipProjectionData.projectionToNDC[0][0] * globals.resolution.x)));

        if (objType == ObjectType::LINE)
        {
            double2 points[2u];
            points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
            points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(LinePointInfo), 8u);

            const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2), 8u);
            const float patternStretch = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) + sizeof(float), 8u);
            outV.setCurrentPhaseShift(phaseShift);
            outV.setPatternStretch(patternStretch);

            float2 transformedPoints[2u];
            for (uint i = 0u; i < 2u; ++i)
            {
                transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, points[i]);
            }

            const float2 lineVector = normalize(transformedPoints[1u] - transformedPoints[0u]);
            const float2 normalToLine = float2(-lineVector.y, lineVector.x);

            if (vertexIdx == 0u || vertexIdx == 1u)
            {
                // work in screen space coordinates because of fixed pixel size
                outV.position.xy = transformedPoints[0u]
                    + normalToLine * (((float)vertexIdx - 0.5f) * 2.0f * antiAliasedLineThickness)
                    - lineVector * antiAliasedLineThickness;
            }
            else // if (vertexIdx == 2u || vertexIdx == 3u)
            {
                // work in screen space coordinates because of fixed pixel size
                outV.position.xy = transformedPoints[1u]
                    + normalToLine * (((float)vertexIdx - 2.5f) * 2.0f * antiAliasedLineThickness)
                    + lineVector * antiAliasedLineThickness;
            }

            outV.setLineStart(transformedPoints[0u]);
            outV.setLineEnd(transformedPoints[1u]);

            outV.position = transformFromSreenSpaceToNdc(outV.position.xy);
        }
        else if (objType == ObjectType::QUAD_BEZIER)
        {
            double2 points[3u];
            points[0u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
            points[1u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
            points[2u] = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2) * 2u, 8u);

            const float phaseShift = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) * 3u, 8u);
            const float patternStretch = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) * 3u + sizeof(float), 8u);
            outV.setCurrentPhaseShift(phaseShift);
            outV.setPatternStretch(patternStretch);

            // transform these points into screen space and pass to fragment
            float2 transformedPoints[3u];
            for (uint i = 0u; i < 3u; ++i)
            {
                transformedPoints[i] = transformPointScreenSpace(clipProjectionData.projectionToNDC, points[i]);
            }

            nbl::hlsl::shapes::QuadraticBezier<float> quadraticBezier = nbl::hlsl::shapes::QuadraticBezier<float>::construct(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u]);
            nbl::hlsl::shapes::Quadratic<float> quadratic = nbl::hlsl::shapes::Quadratic<float>::constructFromBezier(quadraticBezier);
            nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator preCompData = nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator::construct(quadratic);

            outV.setQuadratic(quadratic);
            outV.setQuadraticPrecomputedArcLenData(preCompData);

            float2 Mid = (transformedPoints[0u] + transformedPoints[2u]) / 2.0f;
            float Radius = length(Mid - transformedPoints[0u]) / 2.0f;

            // https://algorithmist.wordpress.com/2010/12/01/quad-bezier-curvature/
            float2 vectorAB = transformedPoints[1u] - transformedPoints[0u];
            float2 vectorAC = transformedPoints[2u] - transformedPoints[1u];
            float area = abs(vectorAB.x * vectorAC.y - vectorAB.y * vectorAC.x) * 0.5;
            float MaxCurvature;
            if (length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.25f)) > Radius && length(transformedPoints[1u] - lerp(transformedPoints[0u], transformedPoints[2u], 0.75f)) > Radius)
                MaxCurvature = pow(length(transformedPoints[1u] - Mid), 3) / (area * area);
            else
                MaxCurvature = max(area / pow(length(transformedPoints[0u] - transformedPoints[1u]), 3), area / pow(length(transformedPoints[2u] - transformedPoints[1u]), 3));

            // We only do this adaptive thing when "MinRadiusOfOsculatingCircle = RadiusOfMaxCurvature < screenSpaceLineWidth/4" OR "MaxCurvature > 4/screenSpaceLineWidth";
            //  which means there is a self intersection because of large lineWidth relative to the curvature (in screenspace)
            //  the reason for division by 4.0f is 1. screenSpaceLineWidth is expanded on both sides and 2. the fact that diameter/2=radius, 
            const bool noCurvature = abs(dot(normalize(vectorAB), normalize(vectorAC)) - 1.0f) < exp2(-10.0f);
            if (MaxCurvature * screenSpaceLineWidth > 4.0f || noCurvature)
            {
                //OBB Fallback
                float2 obbV0;
                float2 obbV1;
                float2 obbV2;
                float2 obbV3;
                quadraticBezier.computeOBB(antiAliasedLineThickness, obbV0, obbV1, obbV2, obbV3);
                if (subsectionIdx == 0)
                {
                    if (vertexIdx == 0u)
                        outV.position = float4(obbV0, 0.0, 1.0f);
                    else if (vertexIdx == 1u)
                        outV.position = float4(obbV1, 0.0, 1.0f);
                    else if (vertexIdx == 2u)
                        outV.position = float4(obbV3, 0.0, 1.0f);
                    else if (vertexIdx == 3u)
                        outV.position = float4(obbV2, 0.0, 1.0f);
                }
                else
                    outV.position = float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            else
            {
                // this optimal value is hardcoded based on tests and benchmarks of pixel shader invocation
                // this is the place where we use it's tangent in the bezier to form sides the cages
                const float optimalT = 0.145f;

                //Whether or not to flip the the interior cage nodes
                int flip = cross2D(transformedPoints[0u] - transformedPoints[1u], transformedPoints[2u] - transformedPoints[1u]) > 0.0f ? -1 : 1;

                const float middleT = 0.5f;
                float2 midPos = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], middleT);
                float2 midTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], middleT));
                float2 midNormal = float2(-midTangent.y, midTangent.x) * flip;

                /*
                            P1
                            +


               exterior0              exterior1
                  ----------------------
                 /                      \-
               -/    ----------------     \
              /    -/interior0     interior1
             /    /                    \    \-
           -/   -/                      \-    \
          /   -/                          \    \-
         /   /                             \-    \
     P0 +                                    \    + P2
                */

                //Internal cage points
                float2 interior0;
                float2 interior1;

                float2 middleExteriorPoint = midPos - midNormal * antiAliasedLineThickness;


                float2 leftTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT));
                float2 leftNormal = normalize(float2(-leftTangent.y, leftTangent.x)) * flip;
                float2 leftExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], optimalT) - leftNormal * antiAliasedLineThickness;
                float2 exterior0 = nbl::hlsl::shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, leftExteriorPoint, leftTangent);

                float2 rightTangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT));
                float2 rightNormal = normalize(float2(-rightTangent.y, rightTangent.x)) * flip;
                float2 rightExteriorPoint = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 1.0f - optimalT) - rightNormal * antiAliasedLineThickness;
                float2 exterior1 = nbl::hlsl::shapes::util::LineLineIntersection<float>(middleExteriorPoint, midTangent, rightExteriorPoint, rightTangent);

                // Interiors
                {
                    float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286f));
                    float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                    interior0 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.286) + normal * antiAliasedLineThickness;
                }
                {
                    float2 tangent = normalize(BezierTangent(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f));
                    float2 normal = normalize(float2(-tangent.y, tangent.x)) * flip;
                    interior1 = QuadraticBezier(transformedPoints[0u], transformedPoints[1u], transformedPoints[2u], 0.714f) + normal * antiAliasedLineThickness;
                }

                if (subsectionIdx == 0u)
                {
                    float2 endPointTangent = normalize(transformedPoints[1u] - transformedPoints[0u]);
                    float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                    float2 endPointExterior = transformedPoints[0u] - endPointTangent * antiAliasedLineThickness;

                    if (vertexIdx == 0u)
                        outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection<float>(leftExteriorPoint, leftTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                    else if (vertexIdx == 1u)
                        outV.position = float4(transformedPoints[0u] + endPointNormal * antiAliasedLineThickness - endPointTangent * antiAliasedLineThickness, 0.0, 1.0f);
                    else if (vertexIdx == 2u)
                        outV.position = float4(exterior0, 0.0, 1.0f);
                    else if (vertexIdx == 3u)
                        outV.position = float4(interior0, 0.0, 1.0f);
                }
                else if (subsectionIdx == 1u)
                {
                    if (vertexIdx == 0u)
                        outV.position = float4(exterior0, 0.0, 1.0f);
                    else if (vertexIdx == 1u)
                        outV.position = float4(interior0, 0.0, 1.0f);
                    else if (vertexIdx == 2u)
                        outV.position = float4(exterior1, 0.0, 1.0f);
                    else if (vertexIdx == 3u)
                        outV.position = float4(interior1, 0.0, 1.0f);
                }
                else if (subsectionIdx == 2u)
                {
                    float2 endPointTangent = normalize(transformedPoints[2u] - transformedPoints[1u]);
                    float2 endPointNormal = float2(-endPointTangent.y, endPointTangent.x) * flip;
                    float2 endPointExterior = transformedPoints[2u] + endPointTangent * antiAliasedLineThickness;

                    if (vertexIdx == 0u)
                        outV.position = float4(nbl::hlsl::shapes::util::LineLineIntersection<float>(rightExteriorPoint, rightTangent, endPointExterior, endPointNormal), 0.0, 1.0f);
                    else if (vertexIdx == 1u)
                        outV.position = float4(transformedPoints[2u] + endPointNormal * antiAliasedLineThickness + endPointTangent * antiAliasedLineThickness, 0.0, 1.0f);
                    else if (vertexIdx == 2u)
                        outV.position = float4(exterior1, 0.0, 1.0f);
                    else if (vertexIdx == 3u)
                        outV.position = float4(interior1, 0.0, 1.0f);
                }
            }

            outV.position.xy = (outV.position.xy / globals.resolution) * 2.0 - 1.0;
        }
        else if (objType == ObjectType::POLYLINE_CONNECTOR)
        {
            const float FLOAT_INF = nbl::hlsl::numeric_limits<float>::infinity;
            const float4 INVALID_VERTEX = float4(FLOAT_INF, FLOAT_INF, FLOAT_INF, FLOAT_INF);

            if (lineStyle.isRoadStyleFlag)
            {
                const double2 circleCenter = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
                const float2 v = vk::RawBufferLoad<float2>(drawObj.geometryAddress + sizeof(double2), 8u);
                const float cosHalfAngleBetweenNormals = vk::RawBufferLoad<float>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 8u);

                const float2 circleCenterScreenSpace = transformPointScreenSpace(clipProjectionData.projectionToNDC, circleCenter);
                outV.setPolylineConnectorCircleCenter(circleCenterScreenSpace);

                // Find other miter vertices
                const float sinHalfAngleBetweenNormals = sqrt(1.0f - (cosHalfAngleBetweenNormals * cosHalfAngleBetweenNormals));
                const float32_t2x2 rotationMatrix = float32_t2x2(cosHalfAngleBetweenNormals, -sinHalfAngleBetweenNormals, sinHalfAngleBetweenNormals, cosHalfAngleBetweenNormals);

                // Pass the precomputed trapezoid values for the sdf
                {
                    float vLen = length(v);
                    float2 intersectionDirection = v / vLen;

                    float longBase = sinHalfAngleBetweenNormals;
                    float shortBase = max((vLen - globals.miterLimit) * cosHalfAngleBetweenNormals / sinHalfAngleBetweenNormals, 0.0);
                    // height of the trapezoid / triangle
                    float hLen = min(globals.miterLimit, vLen);

                    outV.setPolylineConnectorTrapezoidStart(-1.0 * intersectionDirection * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidEnd(intersectionDirection * hLen * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidLongBase(sinHalfAngleBetweenNormals * ((1.0 + vLen) / (vLen - cosHalfAngleBetweenNormals)) * sdfLineThickness);
                    outV.setPolylineConnectorTrapezoidShortBase(shortBase * sdfLineThickness);
                }

                if (vertexIdx == 0u)
                {
                    const float2 V1 = normalize(mul(v, rotationMatrix)) * antiAliasedLineThickness * 2.0f;
                    const float2 screenSpaceV1 = circleCenterScreenSpace + V1;
                    outV.position = float4(screenSpaceV1, 0.0f, 1.0f);   
                }
                else if (vertexIdx == 1u)
                {
                    outV.position = float4(circleCenterScreenSpace, 0.0f, 1.0f);
                }
                else if (vertexIdx == 2u)
                {
                    // find intersection point vertex
                    float2 intersectionPoint = v * antiAliasedLineThickness * 2.0f;
                    intersectionPoint += circleCenterScreenSpace;
                    outV.position = float4(intersectionPoint, 0.0f, 1.0f);
                }
                else if (vertexIdx == 3u)
                {
                    const float2 V2 = normalize(mul(rotationMatrix, v)) * antiAliasedLineThickness * 2.0f;
                    const float2 screenSpaceV2 = circleCenterScreenSpace + V2;
                    outV.position = float4(screenSpaceV2, 0.0f, 1.0f);
                }

                outV.position = transformFromSreenSpaceToNdc(outV.position.xy);
            }
            else
            {
                outV.position = INVALID_VERTEX;
            }
        }
    }
    else if (objType == ObjectType::CURVE_BOX)
    {
        CurveBox curveBox;
        curveBox.aabbMin = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        curveBox.aabbMax = vk::RawBufferLoad<double2>(drawObj.geometryAddress + sizeof(double2), 8u);
        for (uint32_t i = 0; i < 3; i ++)
        {
            curveBox.curveMin[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * i, 4u);
            curveBox.curveMax[i] = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2) * 2 + sizeof(float32_t2) * (3 + i), 4u);
        }

        const float2 ndcAxisU = (float2)transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMax.x, curveBox.aabbMin.y) - curveBox.aabbMin);
        const float2 ndcAxisV = (float2)transformVectorNdc(clipProjectionData.projectionToNDC, double2(curveBox.aabbMin.x, curveBox.aabbMax.y) - curveBox.aabbMin);

        const float2 screenSpaceAabbExtents = float2(length(ndcAxisU * float2(globals.resolution)) / 2.0, length(ndcAxisV * float2(globals.resolution)) / 2.0);

        // we could use something like  this to compute screen space change over minor/major change and avoid ddx(minor), ddy(major) in frag shader (the code below doesn't account for rotation)
        outV.setCurveBoxScreenSpaceSize(float2(screenSpaceAabbExtents));
        
        const float2 undilatedCorner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        
        // We don't dilate on AMD (= no fragShaderInterlock)
        const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
        const float2 dilateRate = pixelsToIncreaseOnEachSide / screenSpaceAabbExtents; // float sufficient to hold the dilate rect? 
        float2 dilateVec;
        float2 dilatedUV;
        dilateHatch<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(dilateVec, dilatedUV, undilatedCorner, dilateRate, ndcAxisU, ndcAxisV);

        // doing interpolation this way to ensure correct endpoints and 0 and 1, we can alternatively use branches to set current corner based on vertexIdx
        const double2 currentCorner = curveBox.aabbMin * (1.0 - undilatedCorner) + curveBox.aabbMax * undilatedCorner;
        const float2 coord = (float2) (transformPointNdc(clipProjectionData.projectionToNDC, currentCorner) + dilateVec);

        outV.position = float4(coord, 0.f, 1.f);
 
        const uint major = (uint)SelectedMajorAxis;
        const uint minor = 1-major;

        // A, B & C get converted from unorm to [0, 1]
        // A & B get converted from [0,1] to [-2, 2]
        nbl::hlsl::shapes::Quadratic<float> curveMin = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMin[0], curveBox.curveMin[1], curveBox.curveMin[2]);
        nbl::hlsl::shapes::Quadratic<float> curveMax = nbl::hlsl::shapes::Quadratic<float>::construct(
            curveBox.curveMax[0], curveBox.curveMax[1], curveBox.curveMax[2]);

        outV.setMinorBBoxUV(dilatedUV[minor]);
        outV.setMajorBBoxUV(dilatedUV[major]);

        outV.setCurveMinMinor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMin.A[minor], 
            curveMin.B[minor], 
            curveMin.C[minor]));
        outV.setCurveMinMajor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMin.A[major], 
            curveMin.B[major], 
            curveMin.C[major]));

        outV.setCurveMaxMinor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMax.A[minor], 
            curveMax.B[minor], 
            curveMax.C[minor]));
        outV.setCurveMaxMajor(nbl::hlsl::math::equations::Quadratic<float>::construct(
            curveMax.A[major], 
            curveMax.B[major], 
            curveMax.C[major]));

        //nbl::hlsl::math::equations::Quadratic<float> curveMinRootFinding = nbl::hlsl::math::equations::Quadratic<float>::construct(
        //    curveMin.A[major], 
        //    curveMin.B[major], 
        //    curveMin.C[major] - maxCorner[major]);
        //nbl::hlsl::math::equations::Quadratic<float> curveMaxRootFinding = nbl::hlsl::math::equations::Quadratic<float>::construct(
        //    curveMax.A[major], 
        //    curveMax.B[major], 
        //    curveMax.C[major] - maxCorner[major]);
        //outV.setMinCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMinRootFinding));
        //outV.setMaxCurvePrecomputedRootFinders(PrecomputedRootFinder<float>::construct(curveMaxRootFinding));
    }
    else if (objType == ObjectType::FONT_GLYPH)
    {
        GlyphInfo glyphInfo;
        glyphInfo.topLeft = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        glyphInfo.dirU = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2), 4u);
        glyphInfo.aspectRatio = vk::RawBufferLoad<float32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 4u);
        glyphInfo.minUV_textureID_packed = vk::RawBufferLoad<uint32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2) + sizeof(float), 4u);

        float32_t2 minUV = glyphInfo.getMinUV();
        uint16_t textureID = glyphInfo.getTextureID();

        const float32_t2 dirV = float32_t2(glyphInfo.dirU.y, -glyphInfo.dirU.x) * glyphInfo.aspectRatio;
        const float2 screenTopLeft = (float2) transformPointNdc(clipProjectionData.projectionToNDC, glyphInfo.topLeft);
        const float2 screenDirU = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, glyphInfo.dirU);
        const float2 screenDirV = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirV);

        const float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1)); // corners of square from (0, 0) to (1, 1)
        const float2 undilatedCornerNDC = corner * 2.0 - 1.0; // corners of square from (-1, -1) to (1, 1)
        
        const float2 screenSpaceAabbExtents = float2(length(screenDirU * float2(globals.resolution)) / 2.0, length(screenDirV * float2(globals.resolution)) / 2.0);
        const float pixelsToIncreaseOnEachSide = globals.antiAliasingFactor + 1.0;
        const float2 dilateRate = (float2)(pixelsToIncreaseOnEachSide / screenSpaceAabbExtents);

        const float2 vx = screenDirU * dilateRate.x;
        const float2 vy = screenDirV * dilateRate.y;
        const float2 offsetVec = vx * undilatedCornerNDC.x + vy * undilatedCornerNDC.y;
        const float2 coord = screenTopLeft + corner.x * screenDirU + corner.y * screenDirV + offsetVec;

        // If aspect ratio of the dimensions and glyph inside the texture are the same then screenPxRangeX === screenPxRangeY
        // but if the glyph box is stretched in any way then we won't get correct msdf
        // in that case we need to take the max(screenPxRangeX, screenPxRangeY) to avoid blur due to underexaggerated distances
        // We compute screenPxRange using the ratio of our screenspace extent to the texel space our glyph takes inside the texture
        // Our glyph is centered inside the texture, so `maxUV = 1.0 - minUV` and `glyphTexelSize = (1.0-2.0*minUV) * MSDFSize
        const float screenPxRangeX = screenSpaceAabbExtents.x / ((1.0 - 2.0 * minUV.x));
        const float screenPxRangeY = screenSpaceAabbExtents.y / ((1.0 - 2.0 * minUV.y));
        float screenPxRange = max(max(screenPxRangeX, screenPxRangeY), 1.0) * MSDFPixelRange / MSDFSize;
        
        // In order to keep the shape scale constant with any dilation values:
        // We compute the new dilated minUV that gets us minUV when interpolated on the previous undilated top left
        const float2 topLeftInterpolationValue = (dilateRate/(1.0+2.0*dilateRate));
        const float2 dilatedMinUV = (topLeftInterpolationValue - minUV) / (2.0 * topLeftInterpolationValue - 1.0);
        const float2 dilatedMaxUV = float2(1.0, 1.0) - dilatedMinUV;
        
        const float2 uv = dilatedMinUV + corner * (dilatedMaxUV - dilatedMinUV);

        outV.position = float4(coord, 0.f, 1.f);
        outV.setFontGlyphUV(uv);
        outV.setFontGlyphTextureId(textureID);
        outV.setFontGlyphScreenPxRange(screenPxRange);
    }
    else if (objType == ObjectType::IMAGE)
    {
        float64_t2 topLeft = vk::RawBufferLoad<double2>(drawObj.geometryAddress, 8u);
        float32_t2 dirU = vk::RawBufferLoad<float32_t2>(drawObj.geometryAddress + sizeof(double2), 4u);
        float32_t aspectRatio = vk::RawBufferLoad<float32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2), 4u);
        uint32_t textureID = vk::RawBufferLoad<uint32_t>(drawObj.geometryAddress + sizeof(double2) + sizeof(float2) + sizeof(float), 4u);

        const float32_t2 dirV = float32_t2(dirU.y, -dirU.x) * aspectRatio;
        const float2 ndcTopLeft = (float2) transformPointNdc(clipProjectionData.projectionToNDC, topLeft);
        const float2 ndcDirU = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirU);
        const float2 ndcDirV = (float2) transformVectorNdc(clipProjectionData.projectionToNDC, dirV);

        float2 corner = float2(bool2(vertexIdx & 0x1u, vertexIdx >> 1));
        float2 uv = corner; // non-dilated
        
        float2 ndcCorner = ndcTopLeft + corner.x * ndcDirU + corner.y * ndcDirV;
        
        outV.position = float4(ndcCorner, 0.f, 1.f);
        outV.setImageUV(uv);
        outV.setImageTextureId(textureID);
    }


// Make the cage fullscreen for testing: 
#if 0
    // disabled for object of POLYLINE_CONNECTOR type, since miters would cover whole screen
    if(objType != ObjectType::POLYLINE_CONNECTOR)
    {
        if (vertexIdx == 0u)
            outV.position = float4(-1, -1, 0, 1);
        else if (vertexIdx == 1u)
            outV.position = float4(-1, +1, 0, 1);
        else if (vertexIdx == 2u)
            outV.position = float4(+1, -1, 0, 1);
        else if (vertexIdx == 3u)
            outV.position = float4(+1, +1, 0, 1);
    }
#endif

    outV.clip = float4(outV.position.x - clipProjectionData.minClipNDC.x, outV.position.y - clipProjectionData.minClipNDC.y, clipProjectionData.maxClipNDC.x - outV.position.x, clipProjectionData.maxClipNDC.y - outV.position.y);
    return outV;
}


// ------- Pixel Shader -------

template<typename float_t>
struct DefaultClipper
{
    using float_t2 = vector<float_t, 2>;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.0;

    static DefaultClipper construct()
    {
        DefaultClipper ret;
        return ret;
    }

    inline float_t2 operator()(const float_t t)
    {
        const float_t ret = clamp(t, 0.0, 1.0);
        return float_t2(ret, ret);
    }
};

// for usage in upper_bound function
struct StyleAccessor
{
    LineStyle style;
    using value_type = float;

    float operator[](const uint32_t ix)
    {
        return style.getStippleValue(ix);
    }
};

template<typename CurveType>
struct StyleClipper
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;
    NBL_CONSTEXPR_STATIC_INLINE float_t AccuracyThresholdT = 0.000001;

    static StyleClipper<CurveType> construct(
        LineStyle style,
        CurveType curve,
        typename CurveType::ArcLengthCalculator arcLenCalc,
        float phaseShift,
        float stretch,
        float worldToScreenRatio)
    {
        StyleClipper<CurveType> ret = { style, curve, arcLenCalc, phaseShift, stretch, worldToScreenRatio, 0.0f, 0.0f, 0.0f, 0.0f };

        // values for non-uniform stretching with a rigid segment
        if (style.rigidSegmentIdx != InvalidRigidSegmentIndex && stretch != 1.0f)
        {
            // rigidSegment info in old non stretched pattern
            ret.rigidSegmentStart = (style.rigidSegmentIdx >= 1u) ? style.getStippleValue(style.rigidSegmentIdx - 1u) : 0.0f;
            ret.rigidSegmentEnd = (style.rigidSegmentIdx < style.stipplePatternSize) ? style.getStippleValue(style.rigidSegmentIdx) : 1.0f;
            ret.rigidSegmentLen = ret.rigidSegmentEnd - ret.rigidSegmentStart;
            // stretch value for non rigid segments
            ret.nonRigidSegmentStretchValue = (stretch - ret.rigidSegmentLen) / (1.0f - ret.rigidSegmentLen);
            // rigidSegment info to new stretched pattern
            ret.rigidSegmentStart *= ret.nonRigidSegmentStretchValue / stretch; // get the new normalized rigid segment start
            ret.rigidSegmentLen /= stretch; // get the new rigid segment normalized len
            ret.rigidSegmentEnd = ret.rigidSegmentStart + ret.rigidSegmentLen; // get the new normalized rigid segment end 
        }
        else
        {
            ret.nonRigidSegmentStretchValue = stretch;
        }
        
        return ret;
    }

    // For non-uniform stretching with a rigid segment (the one segement that shouldn't stretch) the whole pattern changes
    // instead of transforming each of the style.stipplePattern values (max 14 of them), we transform the normalized place in pattern
    float getRealNormalizedPlaceInPattern(float normalizedPlaceInPattern)
    {
        if (style.rigidSegmentIdx != InvalidRigidSegmentIndex && stretch != 1.0f)
        {
            float ret = min(normalizedPlaceInPattern, rigidSegmentStart) / nonRigidSegmentStretchValue; // unstretch parts before rigid segment
            ret += max(normalizedPlaceInPattern - rigidSegmentEnd, 0.0f) / nonRigidSegmentStretchValue; // unstretch parts after rigid segment
            ret += max(min(rigidSegmentLen, normalizedPlaceInPattern - rigidSegmentStart), 0.0f); // unstretch parts inside rigid segment
            ret *= stretch;
            return ret;
        }
        else
        {
            return normalizedPlaceInPattern;
        }
    }

    float_t2 operator()(float_t t)
    {
        // basicaly 0.0 and 1.0 but with a guardband to discard outside the range
        const float_t minT = 0.0 - 1.0;
        const float_t maxT = 1.0 + 1.0;

        StyleAccessor styleAccessor = { style };
        const float_t reciprocalStretchedStipplePatternLen = style.reciprocalStipplePatternLen / stretch;
        const float_t patternLenInScreenSpace = 1.0 / (worldToScreenRatio * style.reciprocalStipplePatternLen);

        const float_t arcLen = arcLenCalc.calcArcLen(t);
        const float_t worldSpaceArcLen = arcLen * float_t(worldToScreenRatio);
        float_t normalizedPlaceInPattern = frac(worldSpaceArcLen * reciprocalStretchedStipplePatternLen + phaseShift);
        normalizedPlaceInPattern = getRealNormalizedPlaceInPattern(normalizedPlaceInPattern);
        uint32_t patternIdx = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPattern);

        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::infinity; 
        float_t2 ret = float_t2(InvalidT, InvalidT);

        // odd patternIdx means a "no draw section" and current candidate should split into two nearest draw sections
        const bool notInDrawSection = patternIdx & 0x1;
        
        // TODO[Erfan]: Disable this piece of code after clipping, and comment the reason, that the bezier start and end at 0.0 and 1.0 should be in drawable sections
        float_t minDrawT = 0.0;
        float_t maxDrawT = 1.0;
        {
            float_t normalizedPlaceInPatternBegin = frac(phaseShift);
            normalizedPlaceInPatternBegin = getRealNormalizedPlaceInPattern(normalizedPlaceInPatternBegin);
            uint32_t patternIdxBegin = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternBegin);
            const bool BeginInNonDrawSection = patternIdxBegin & 0x1;

            if (BeginInNonDrawSection)
            {
                float_t diffToRightDrawableSection = (patternIdxBegin == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdxBegin];
                diffToRightDrawableSection -= normalizedPlaceInPatternBegin;
                float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * patternLenInScreenSpace * ((patternIdxBegin != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);
                const float_t arcLenForT1 = 0.0 + scrSpcOffsetToArcLen1;
                minDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, 0.0);
            }
            
            // Completely in non-draw section -> clip away:
            if (minDrawT >= 1.0)
                return ret;

            const float_t arcLenEnd = arcLenCalc.calcArcLen(1.0);
            const float_t worldSpaceArcLenEnd = arcLenEnd * float_t(worldToScreenRatio);
            float_t normalizedPlaceInPatternEnd = frac(worldSpaceArcLenEnd * reciprocalStretchedStipplePatternLen + phaseShift);
            normalizedPlaceInPatternEnd = getRealNormalizedPlaceInPattern(normalizedPlaceInPatternEnd);
            uint32_t patternIdxEnd = nbl::hlsl::upper_bound(styleAccessor, 0, style.stipplePatternSize, normalizedPlaceInPatternEnd);
            const bool EndInNonDrawSection = patternIdxEnd & 0x1;

            if (EndInNonDrawSection)
            {
                float_t diffToLeftDrawableSection = (patternIdxEnd == 0) ? 0.0 : styleAccessor[patternIdxEnd - 1];
                diffToLeftDrawableSection -= normalizedPlaceInPatternEnd;
                float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * patternLenInScreenSpace * ((patternIdxEnd != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);
                const float_t arcLenForT0 = arcLenEnd + scrSpcOffsetToArcLen0;
                maxDrawT = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, 1.0);
            }
        }

        if (notInDrawSection)
        {
            float toScreenSpaceLen = patternLenInScreenSpace * ((patternIdx != style.rigidSegmentIdx) ? nonRigidSegmentStretchValue : 1.0);

            float_t diffToLeftDrawableSection = (patternIdx == 0) ? 0.0 : styleAccessor[patternIdx - 1];
            diffToLeftDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen0 = diffToLeftDrawableSection * toScreenSpaceLen;
            const float_t arcLenForT0 = arcLen + scrSpcOffsetToArcLen0;
            float_t t0 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT0, AccuracyThresholdT, t);
            t0 = clamp(t0, minDrawT, maxDrawT);

            float_t diffToRightDrawableSection = (patternIdx == style.stipplePatternSize) ? 1.0 : styleAccessor[patternIdx];
            diffToRightDrawableSection -= normalizedPlaceInPattern;
            float_t scrSpcOffsetToArcLen1 = diffToRightDrawableSection * toScreenSpaceLen;
            const float_t arcLenForT1 = arcLen + scrSpcOffsetToArcLen1;
            float_t t1 = arcLenCalc.calcArcLenInverse(curve, minT, maxT, arcLenForT1, AccuracyThresholdT, t);
            t1 = clamp(t1, minDrawT, maxDrawT);

            ret = float_t2(t0, t1);
        }
        else
        {
            t = clamp(t, minDrawT, maxDrawT);
            ret = float_t2(t, t);
        }

        return ret;
    }

    LineStyle style;
    CurveType curve;
    typename CurveType::ArcLengthCalculator arcLenCalc;
    float phaseShift;
    float stretch;
    float worldToScreenRatio;
    // precomp value for non uniform stretching
    float rigidSegmentStart;
    float rigidSegmentEnd;
    float rigidSegmentLen;
    float nonRigidSegmentStretchValue;
};

template<typename CurveType, typename Clipper = DefaultClipper<typename CurveType::scalar_t> >
struct ClippedSignedDistance
{
    using float_t = typename CurveType::scalar_t;
    using float_t2 = typename CurveType::float_t2;
    using float_t3 = typename CurveType::float_t3;

    const static float_t sdf(CurveType curve, float_t2 pos, float_t thickness, bool isRoadStyle, Clipper clipper = DefaultClipper<typename CurveType::scalar_t>::construct())
    {
        typename CurveType::Candidates candidates = curve.getClosestCandidates(pos);

        const float_t InvalidT = nbl::hlsl::numeric_limits<float32_t>::max;
        // TODO: Fix and test, we're not working with squared distance anymore
        const float_t MAX_DISTANCE_SQUARED = (thickness + 1.0f) * (thickness + 1.0f); // TODO: ' + 1' is too much?

        bool clipped = false;
        float_t closestDistanceSquared = MAX_DISTANCE_SQUARED;
        float_t closestT = InvalidT;
        [[unroll(CurveType::MaxCandidates)]]
        for (uint32_t i = 0; i < CurveType::MaxCandidates; i++)
        {
            const float_t candidateDistanceSquared = length(curve.evaluate(candidates[i]) - pos);
            if (candidateDistanceSquared < closestDistanceSquared)
            {
                float_t2 snappedTs = clipper(candidates[i]);

                if (snappedTs[0] == InvalidT)
                {
                    continue;
                }

                if (snappedTs[0] != candidates[i])
                {
                    // left snapped or clamped
                    const float_t leftSnappedCandidateDistanceSquared = length(curve.evaluate(snappedTs[0]) - pos);
                    if (leftSnappedCandidateDistanceSquared < closestDistanceSquared)
                    {
                        clipped = true;
                        closestT = snappedTs[0];
                        closestDistanceSquared = leftSnappedCandidateDistanceSquared;
                    }

                    if (snappedTs[0] != snappedTs[1])
                    {
                        // right snapped or clamped
                        const float_t rightSnappedCandidateDistanceSquared = length(curve.evaluate(snappedTs[1]) - pos);
                        if (rightSnappedCandidateDistanceSquared < closestDistanceSquared)
                        {
                            clipped = true;
                            closestT = snappedTs[1];
                            closestDistanceSquared = rightSnappedCandidateDistanceSquared;
                        }
                    }
                }
                else
                {
                    // no snapping
                    if (candidateDistanceSquared < closestDistanceSquared)
                    {
                        clipped = false;
                        closestT = candidates[i];
                        closestDistanceSquared = candidateDistanceSquared;
                    }
                }
            }
        }


        float_t roundedDistance = closestDistanceSquared - thickness;
        if(!isRoadStyle)
        {
            return roundedDistance;
        }
        else
        {
            const float_t aaWidth = globals.antiAliasingFactor;
            float_t rectCappedDistance = roundedDistance;

            if (clipped)
            {
                float_t2 q = mul(curve.getLocalCoordinateSpace(closestT), pos - curve.evaluate(closestT));
                rectCappedDistance = capSquare(q, thickness, aaWidth);
            }

            return rectCappedDistance;
        }
    }

    static float capSquare(float_t2 q, float_t th, float_t aaWidth)
    {
        float_t2 d = abs(q) - float_t2(aaWidth, th);
        return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
    }
};

// sdf of Isosceles Trapezoid y-aligned by https://iquilezles.org/articles/distfunctions2d/
float sdTrapezoid(float2 p, float r1, float r2, float he)
{
    float2 k1 = float2(r2, he);
    float2 k2 = float2(r2 - r1, 2.0 * he);

    p.x = abs(p.x);
    float2 ca = float2(max(0.0, p.x - ((p.y < 0.0) ? r1 : r2)), abs(p.y) - he);
    float2 cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2,k2), 0.0, 1.0);

    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;

    return s * sqrt(min(dot(ca,ca), dot(cb,cb)));
}

// line segment sdf which returns the distance vector specialized for usage in hatch box line boundaries
float2 sdLineDstVec(float2 P, float2 A, float2 B)
{
    const float2 PA = P - A;
    const float2 BA = B - A;
    float h = clamp(dot(PA, BA) / dot(BA, BA), 0.0, 1.0);
    return PA - BA * h;
}

float miterSDF(float2 p, float thickness, float2 a, float2 b, float ra, float rb)
{
    float h = length(b - a) / 2.0;
    float2 d = normalize(b - a);
    float2x2 rot = float2x2(d.y, -d.x, d.x, d.y);
    p = mul(rot, p);
    p.y -= h - thickness;
    return sdTrapezoid(p, ra, rb, h);
}

typedef StyleClipper< nbl::hlsl::shapes::Quadratic<float> > BezierStyleClipper;
typedef StyleClipper< nbl::hlsl::shapes::Line<float> > LineStyleClipper;

// We need to specialize color calculation based on FragmentShaderInterlock feature availability for our transparency algorithm
// because there is no `if constexpr` in hlsl
// @params
// textureColor: color sampled from a texture
// useStyleColor: instead of writing and reading from colorStorage, use main object Idx to find the style color for the object.
template<bool FragmentShaderPixelInterlock>
float32_t4 calculateFinalColor(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 textureColor);

template<>
float32_t4 calculateFinalColor<false>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor)
{
    uint32_t styleIdx = mainObjects[currentMainObjectIdx].styleIdx;
    const bool colorFromStyle = styleIdx != InvalidStyleIdx;
    if (colorFromStyle)
    {
        float32_t4 col = lineStyles[styleIdx].color;
        col.w *= localAlpha;
        return float4(col);
    }
    else
        return float4(localTextureColor, localAlpha);
}
template<>
float32_t4 calculateFinalColor<true>(const uint2 fragCoord, const float localAlpha, const uint32_t currentMainObjectIdx, float3 localTextureColor)
{
    float32_t4 color;
    
    nbl::hlsl::spirv::execution_mode::PixelInterlockOrderedEXT();
    nbl::hlsl::spirv::beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];

    const uint32_t localQuantizedAlpha = (uint32_t)(localAlpha * 255.f);
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);
    // if geomID has changed, we resolve the SDF alpha (draw using blend), else accumulate
    const bool resolve = currentMainObjectIdx != storedMainObjectIdx;
    uint32_t resolveStyleIdx = mainObjects[storedMainObjectIdx].styleIdx;
    const bool resolveColorFromStyle = resolveStyleIdx != InvalidStyleIdx;
    
    // load from colorStorage only if we want to resolve color from texture instead of style
    // sampling from colorStorage needs to happen in critical section because another fragment may also want to store into it at the same time + need to happen before store
    if (resolve && !resolveColorFromStyle)
        color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);

    if (resolve || localQuantizedAlpha > storedQuantizedAlpha)
    {
        pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(localQuantizedAlpha,currentMainObjectIdx,AlphaBits,MainObjectIdxBits);
        colorStorage[fragCoord] = packR11G11B10_UNORM(localTextureColor);
    }
    
    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (!resolve)
        discard;

    // draw with previous geometry's style's color or stored in texture buffer :kek:
    // we don't need to load the style's color in critical section because we've already retrieved the style index from the stored main obj
    if (resolveColorFromStyle)
        color = lineStyles[resolveStyleIdx].color;
    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}


[shader("pixel")]
float4 ps_main(PSInput input) : SV_TARGET
{
    float localAlpha = 0.0f;
    ObjectType objType = input.getObjType();
    const uint32_t currentMainObjectIdx = input.getMainObjectIdx();
    const MainObject mainObj = mainObjects[currentMainObjectIdx];
    float3 textureColor = float3(0, 0, 0); // color sampled from a texture
    
    // figure out local alpha with sdf
    if (objType == ObjectType::LINE || objType == ObjectType::QUAD_BEZIER || objType == ObjectType::POLYLINE_CONNECTOR)
    {
        float distance = nbl::hlsl::numeric_limits<float>::max;
        if (objType == ObjectType::LINE)
        {
            const float2 start = input.getLineStart();
            const float2 end = input.getLineEnd();
            const uint32_t styleIdx = mainObj.styleIdx;
            const float thickness = input.getLineThickness();
            const float phaseShift = input.getCurrentPhaseShift();
            const float stretch = input.getPatternStretch();
            const float worldToScreenRatio = input.getCurrentWorldToScreenRatio();

            nbl::hlsl::shapes::Line<float> lineSegment = nbl::hlsl::shapes::Line<float>::construct(start, end);
            nbl::hlsl::shapes::Line<float>::ArcLengthCalculator arcLenCalc = nbl::hlsl::shapes::Line<float>::ArcLengthCalculator::construct(lineSegment);

            LineStyle style = lineStyles[styleIdx];

            if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
            {
                distance = ClippedSignedDistance< nbl::hlsl::shapes::Line<float> >::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag);
            }
            else
            {
                LineStyleClipper clipper = LineStyleClipper::construct(lineStyles[styleIdx], lineSegment, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
                distance = ClippedSignedDistance<nbl::hlsl::shapes::Line<float>, LineStyleClipper>::sdf(lineSegment, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
            }
        }
        else if (objType == ObjectType::QUAD_BEZIER)
        {
            nbl::hlsl::shapes::Quadratic<float> quadratic = input.getQuadratic();
            nbl::hlsl::shapes::Quadratic<float>::ArcLengthCalculator arcLenCalc = input.getQuadraticArcLengthCalculator();

            const uint32_t styleIdx = mainObj.styleIdx;
            const float thickness = input.getLineThickness();
            const float phaseShift = input.getCurrentPhaseShift();
            const float stretch = input.getPatternStretch();
            const float worldToScreenRatio = input.getCurrentWorldToScreenRatio();

            LineStyle style = lineStyles[styleIdx];
            if (!style.hasStipples() || stretch == InvalidStyleStretchValue)
            {
                distance = ClippedSignedDistance< nbl::hlsl::shapes::Quadratic<float> >::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag);
            }
            else
            {
                BezierStyleClipper clipper = BezierStyleClipper::construct(lineStyles[styleIdx], quadratic, arcLenCalc, phaseShift, stretch, worldToScreenRatio);
                distance = ClippedSignedDistance<nbl::hlsl::shapes::Quadratic<float>, BezierStyleClipper>::sdf(quadratic, input.position.xy, thickness, style.isRoadStyleFlag, clipper);
            }
        }
        else if (objType == ObjectType::POLYLINE_CONNECTOR)
        {
            const float2 P = input.position.xy - input.getPolylineConnectorCircleCenter();
            distance = miterSDF(
                P,
                input.getLineThickness(),
                input.getPolylineConnectorTrapezoidStart(),
                input.getPolylineConnectorTrapezoidEnd(),
                input.getPolylineConnectorTrapezoidLongBase(),
                input.getPolylineConnectorTrapezoidShortBase());

        }
        localAlpha = smoothstep(+globals.antiAliasingFactor, -globals.antiAliasingFactor, distance);
    }
    else if (objType == ObjectType::CURVE_BOX) 
    {
        const float minorBBoxUV = input.getMinorBBoxUV();
        const float majorBBoxUV = input.getMajorBBoxUV();

        nbl::hlsl::math::equations::Quadratic<float> curveMinMinor = input.getCurveMinMinor();
        nbl::hlsl::math::equations::Quadratic<float> curveMinMajor = input.getCurveMinMajor();
        nbl::hlsl::math::equations::Quadratic<float> curveMaxMinor = input.getCurveMaxMinor();
        nbl::hlsl::math::equations::Quadratic<float> curveMaxMajor = input.getCurveMaxMajor();

        //  TODO(Optimization): Can we ignore this majorBBoxUV clamp and rely on the t clamp that happens next? then we can pass `PrecomputedRootFinder`s instead of computing the values per pixel.
        nbl::hlsl::math::equations::Quadratic<float> minCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMinMajor.a, curveMinMajor.b, curveMinMajor.c - clamp(majorBBoxUV, 0.0, 1.0));
        nbl::hlsl::math::equations::Quadratic<float> maxCurveEquation = nbl::hlsl::math::equations::Quadratic<float>::construct(curveMaxMajor.a, curveMaxMajor.b, curveMaxMajor.c - clamp(majorBBoxUV, 0.0, 1.0));

        const float minT = clamp(PrecomputedRootFinder<float>::construct(minCurveEquation).computeRoots(), 0.0, 1.0);
        const float minEv = curveMinMinor.evaluate(minT);

        const float maxT = clamp(PrecomputedRootFinder<float>::construct(maxCurveEquation).computeRoots(), 0.0, 1.0);
        const float maxEv = curveMaxMinor.evaluate(maxT);

        const bool insideMajor = majorBBoxUV >= 0.0 && majorBBoxUV <= 1.0;
        const bool insideMinor = minorBBoxUV >= minEv && minorBBoxUV <= maxEv;

        if (insideMinor && insideMajor)
        {
            localAlpha = 1.0;
        }
        else
        {
            // Find the true SDF of a hatch box boundary which is bounded by two curves, It requires knowing the distance from the current UV to the closest point on bounding curves and the limiting lines (in major direction)
            // We also keep track of distance vector (minor, major) to convert to screenspace distance for anti-aliasing with screenspace aaFactor
            const float InvalidT = nbl::hlsl::numeric_limits<float32_t>::max;
            const float MAX_DISTANCE_SQUARED = nbl::hlsl::numeric_limits<float32_t>::max;

            const float2 boxScreenSpaceSize = input.getCurveBoxScreenSpaceSize();


            float closestDistanceSquared = MAX_DISTANCE_SQUARED;
            const float2 pos = float2(minorBBoxUV, majorBBoxUV) * boxScreenSpaceSize;

            if (minorBBoxUV < minEv)
            {
                // DO SDF of Min Curve
                nbl::hlsl::shapes::Quadratic<float> minCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                    float2(curveMinMinor.a, curveMinMajor.a) * boxScreenSpaceSize,
                    float2(curveMinMinor.b, curveMinMajor.b) * boxScreenSpaceSize,
                    float2(curveMinMinor.c, curveMinMajor.c) * boxScreenSpaceSize);

                nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = minCurve.getClosestCandidates(pos);
                [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                {
                    candidates[i] = clamp(candidates[i], 0.0, 1.0);
                    const float2 distVector = minCurve.evaluate(candidates[i]) - pos;
                    const float candidateDistanceSquared = dot(distVector, distVector);
                    if (candidateDistanceSquared < closestDistanceSquared)
                        closestDistanceSquared = candidateDistanceSquared;
                }
            }
            else if (minorBBoxUV > maxEv)
            {
                // Do SDF of Max Curve
                nbl::hlsl::shapes::Quadratic<float> maxCurve = nbl::hlsl::shapes::Quadratic<float>::construct(
                    float2(curveMaxMinor.a, curveMaxMajor.a) * boxScreenSpaceSize,
                    float2(curveMaxMinor.b, curveMaxMajor.b) * boxScreenSpaceSize,
                    float2(curveMaxMinor.c, curveMaxMajor.c) * boxScreenSpaceSize);
                nbl::hlsl::shapes::Quadratic<float>::Candidates candidates = maxCurve.getClosestCandidates(pos);
                [[unroll(nbl::hlsl::shapes::Quadratic<float>::MaxCandidates)]]
                for (uint32_t i = 0; i < nbl::hlsl::shapes::Quadratic<float>::MaxCandidates; i++)
                {
                    candidates[i] = clamp(candidates[i], 0.0, 1.0);
                    const float2 distVector = maxCurve.evaluate(candidates[i]) - pos;
                    const float candidateDistanceSquared = dot(distVector, distVector);
                    if (candidateDistanceSquared < closestDistanceSquared)
                        closestDistanceSquared = candidateDistanceSquared;
                }
            }

            if (!insideMajor)
            {
                const bool minLessThanMax = minEv < maxEv;
                float2 majorDistVector = float2(MAX_DISTANCE_SQUARED, MAX_DISTANCE_SQUARED);
                if (majorBBoxUV > 1.0)
                {
                    const float2 minCurveEnd = float2(minEv, 1.0) * boxScreenSpaceSize;
                    if (minLessThanMax)
                        majorDistVector = sdLineDstVec(pos, minCurveEnd, float2(maxEv, 1.0) * boxScreenSpaceSize);
                    else
                        majorDistVector = pos - minCurveEnd;
                }
                else
                {
                    const float2 minCurveStart = float2(minEv, 0.0) * boxScreenSpaceSize;
                    if (minLessThanMax)
                        majorDistVector = sdLineDstVec(pos, minCurveStart, float2(maxEv, 0.0) * boxScreenSpaceSize);
                    else
                        majorDistVector = pos - minCurveStart;
                }

                const float majorDistSq = dot(majorDistVector, majorDistVector);
                if (majorDistSq < closestDistanceSquared)
                    closestDistanceSquared = majorDistSq;
            }

            const float dist = sqrt(closestDistanceSquared);
            localAlpha = 1.0f - smoothstep(0.0, globals.antiAliasingFactor, dist);
        }

        LineStyle style = lineStyles[mainObj.styleIdx];
        uint32_t textureId = asuint(style.screenSpaceLineWidth);
        if (textureId != InvalidTextureIdx)
        {
            // For Hatch fiils we sample the first mip as we don't fill the others, because they are constant in screenspace and render as expected
            // If later on we decided that we can have different sizes here, we should do computations similar to FONT_GLYPH
            float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(frac(input.position.xy / HatchFillMSDFSceenSpaceSize), float(textureId)), 0.0).xyz;
            float msdf = nbl::hlsl::text::msdfDistance(msdfSample, MSDFPixelRange * HatchFillMSDFSceenSpaceSize / MSDFSize);
            localAlpha *= smoothstep(+globals.antiAliasingFactor / 2.0, -globals.antiAliasingFactor / 2.0f, msdf);
        }
    }
    else if (objType == ObjectType::FONT_GLYPH) 
    {
        const float2 uv = input.getFontGlyphUV();
        const uint32_t textureId = input.getFontGlyphTextureId();

        if (textureId != InvalidTextureIdx)
        {
            float mipLevel = msdfTextures.CalculateLevelOfDetail(msdfSampler, uv);
            float3 msdfSample = msdfTextures.SampleLevel(msdfSampler, float3(uv, float(textureId)), mipLevel);
            float msdf = nbl::hlsl::text::msdfDistance(msdfSample, input.getFontGlyphScreenPxRange());
            /*
                explaining "*= exp2(max(mipLevel,0.0))"
                Each mip level has constant MSDFPixelRange
                Which essentially makes the msdfSamples here (Harware Sampled) have different scales per mip
                As we go up 1 mip level, the msdf distance should be multiplied by 2.0
                While this makes total sense for NEAREST mip sampling when mipLevel is an integer and only one mip is being sampled.
                It's a bit complex when it comes to trilinear filtering (LINEAR mip sampling), but it works in practice!
                
                Alternatively you can think of it as doing this instead:
                localAlpha = smoothstep(+globals.antiAliasingFactor / exp2(max(mipLevel,0.0)), 0.0, msdf);
                Which is reducing the aa feathering as we go up the mip levels. 
                to avoid aa feathering of the MAX_MSDF_DISTANCE_VALUE to be less than aa factor and eventually color it and cause greyed out area around the main glyph
            */
            msdf *= exp2(max(mipLevel,0.0));
            localAlpha = smoothstep(+globals.antiAliasingFactor, 0.0, msdf);
        }
    }
    else if (objType == ObjectType::IMAGE) 
    {
        const float2 uv = input.getImageUV();
        const uint32_t textureId = input.getImageTextureId();

        if (textureId != InvalidTextureIdx)
        {
            float4 colorSample = textures[NonUniformResourceIndex(textureId)].Sample(textureSampler, float2(uv.x, uv.y));
            textureColor = colorSample.rgb;
            localAlpha = colorSample.a;
        }
    }

    uint2 fragCoord = uint2(input.position.xy);
    
    if (localAlpha <= 0)
        discard;
    
    return calculateFinalColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(fragCoord, localAlpha, currentMainObjectIdx, textureColor);
}

// ------- Pixel Debug Shader -------

[shader("pixel")]
float4 ps_debug_main(PSInput input) : SV_TARGET
{
    return float4(1.0, 1.0, 1.0, 1.0);
// return input.color;
}

// ------- Resolve Alpha Shader -------

template<bool FragmentShaderPixelInterlock>
float32_t4 calculateFinalColor(const uint2 fragCoord);


template<>
float32_t4 calculateFinalColor<false>(const uint2 fragCoord)
{
    return float4(0.0f, 0.0f, 0.0f, 0.0f);
}

template<>
float32_t4 calculateFinalColor<true>(const uint2 fragCoord)
{
    float32_t4 color;
    
    nbl::hlsl::spirv::execution_mode::PixelInterlockOrderedEXT();
    nbl::hlsl::spirv::beginInvocationInterlockEXT();

    const uint32_t packedData = pseudoStencil[fragCoord];
    const uint32_t storedQuantizedAlpha = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,0,AlphaBits);
    const uint32_t storedMainObjectIdx = nbl::hlsl::glsl::bitfieldExtract<uint32_t>(packedData,AlphaBits,MainObjectIdxBits);
    pseudoStencil[fragCoord] = nbl::hlsl::glsl::bitfieldInsert<uint32_t>(0, InvalidMainObjectIdx, AlphaBits, MainObjectIdxBits);
    
    uint32_t resolveStyleIdx = mainObjects[storedMainObjectIdx].styleIdx;
    const bool resolveColorFromStyle = resolveStyleIdx != InvalidStyleIdx;
    if (!resolveColorFromStyle)
        color = float32_t4(unpackR11G11B10_UNORM(colorStorage[fragCoord]), 1.0f);

    nbl::hlsl::spirv::endInvocationInterlockEXT();

    if (resolveColorFromStyle)
        color = lineStyles[resolveStyleIdx].color;
    color.a *= float(storedQuantizedAlpha) / 255.f;
    
    return color;
}

[shader("pixel")]
float4 ra_main(float4 position : SV_Position) : SV_TARGET
{
    return calculateFinalColor<nbl::hlsl::jit::device_capabilities::fragmentShaderPixelInterlock>(position.xy);
}
