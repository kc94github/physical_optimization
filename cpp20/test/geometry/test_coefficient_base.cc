#include <gtest/gtest.h>
#include "coefficient_base.h"

TEST(BasicTest, HandlesEquality) {
    Geometry::CoefficientBase a(3);
    EXPECT_EQ(a.order(), 3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}