#include <Symbol.hpp>
#include <cassert>
#include <iostream>
#include <set>
#include <unordered_set>

using namespace chord;

int main() {
    std::cout << "Testing Symbol class...\n";

    Symbol sym("Hello, World!");

    const Symbol x("x");
    const Symbol y("y");
    const Symbol x2("x");

    std::cout << "x: " << x << "\n";
    std::cout << "y: " << y << "\n";
    std::cout << "x2: " << x2 << "\n";

    assert(x != y);
    assert(x == x2);

    std::set<Symbol> set1;
    std::unordered_set<Symbol> set2;

    set1.insert(x);
    set2.insert(x);
    assert(set1.count(x));
    assert(!set1.count(y));
    assert(set2.count(x));
    assert(!set2.count(y));

    std::cout << "Success.\n";
    return 0;
}
