#include <iostream>
#include <string>

// Abstract base class equivalent in C++
class Abstract {
    public:
        virtual ~Abstract() = default;

        virtual std::string toString() const = 0;
};