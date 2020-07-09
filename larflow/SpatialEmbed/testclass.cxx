#include "testclass.h"

namespace larflow {
namespace spatialembed {

testclass::testclass(int what) 
{
    hello = what;
}

testclass::~testclass() 
{
}

int testclass::gethello()
{
    return hello;
}

}
}