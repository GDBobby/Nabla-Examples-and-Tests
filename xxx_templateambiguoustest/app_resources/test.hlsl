#ifndef XXX_TEST_HLSL
#define XXX_TEST_HLSL

#include "nbl/builtin/hlsl/cpp_compat.hlsl"

// shows error: 'nbl::hlsl::dot': ambiguous call to overloaded function


template<typename T>
struct SHLSLThing
{
    using this_t = SHLSLThing<T>;
    using scalar_type = T;
    using vector3_type = vector<T, 3>;

    static this_t create(NBL_CONST_REF_ARG(vector3_type) N, NBL_CONST_REF_ARG(vector3_type) V)
    {
        this_t retval;
        retval.dir = dot<vector3_type>(N, V);
        return retval;
    }

    scalar_type dir;
};

// weird observation:
// if this struct is commented out, then only one compile error on line 17 as expected
// if this struct is compiled as well (not commented out), then only two compile errors on line 32 + 33, no error on line 17
struct SAnotherThing
{
    static SAnotherThing create(NBL_CONST_REF_ARG(float32_t3) N, NBL_CONST_REF_ARG(float32_t3) V)
    {
        SAnotherThing retval;
        float B = dot<float32_t3>(N, float32_t3(0,1,0));
        retval.dir = dot<float32_t3>(N, V);
        return retval;
    }

    float dir;
};

#endif
