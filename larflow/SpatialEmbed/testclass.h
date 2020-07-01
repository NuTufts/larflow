#ifndef __testclass__
#define __testclass__

#include <Python.h>


namespace larflow {
  
namespace spatialembed {

class testclass{
private:
    int hello;

public:
    testclass(int what);
    ~testclass();

    int gethello();

};

}
}