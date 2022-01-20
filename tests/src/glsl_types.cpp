#include <iostream>
using namespace std;

#include <rtac_display/GLSLType.h>
using namespace rtac::display;

using namespace rtac::types;

int main()
{
    cout << GLSLTypeFloat<1>::value << endl;
    cout << GLSLTypeFloat<2>::value << endl;
    cout << GLSLTypeFloat<3>::value << endl;
    cout << GLSLTypeFloat<4>::value << endl;

    cout << GLSLTypeInt<1>::value << endl;
    cout << GLSLTypeInt<2>::value << endl;
    cout << GLSLTypeInt<3>::value << endl;
    cout << GLSLTypeInt<4>::value << endl;

    cout << GLSLTypeUint<1>::value << endl;
    cout << GLSLTypeUint<2>::value << endl;
    cout << GLSLTypeUint<3>::value << endl;
    cout << GLSLTypeUint<4>::value << endl;
    cout << endl << endl;

    cout << GLSLType<float>::value << endl;
    cout << GLSLType<Point2<float>>::value << endl;
    cout << GLSLType<Point3<float>>::value << endl;
    cout << GLSLType<Point4<float>>::value << endl;

    cout << GLSLType<unsigned char>::value << endl;
    cout << GLSLType<Point2<unsigned char>>::value << endl;
    cout << GLSLType<Point3<unsigned char>>::value << endl;
    cout << GLSLType<Point4<unsigned char>>::value << endl;

    cout << GLSLType<uint32_t>::value << endl;
    cout << GLSLType<Point2<uint32_t>>::value << endl;
    cout << GLSLType<Point3<uint32_t>>::value << endl;
    cout << GLSLType<Point4<uint32_t>>::value << endl;

    cout << GLSLType<char>::value << endl;
    cout << GLSLType<Point2<char>>::value << endl;
    cout << GLSLType<Point3<char>>::value << endl;
    cout << GLSLType<Point4<char>>::value << endl;

    cout << GLSLType<int32_t>::value << endl;
    cout << GLSLType<Point2<int32_t>>::value << endl;
    cout << GLSLType<Point3<int32_t>>::value << endl;
    cout << GLSLType<Point4<int32_t>>::value << endl;

    return 0;
}
