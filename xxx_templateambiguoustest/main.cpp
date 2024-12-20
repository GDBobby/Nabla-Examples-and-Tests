#include <nabla.h>
#include <iostream>

#include <nbl/builtin/hlsl/cpp_compat.hlsl>

using namespace nbl::hlsl;

#include "app_resources/test.hlsl"

int main(int argc, char* argv)
{
    SHLSLThing<float> hlslthing = SHLSLThing<float>::create(float32_t3(0.2), float32_t3(0.3));
    std::cout << hlslthing.dir << "\n";

    SAnotherThing anothing = SAnotherThing::create(float32_t3(0.2), float32_t3(0.3));
    std::cout << anothing.dir << "\n";

    return 0;
}