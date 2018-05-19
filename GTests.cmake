include(GoogleTest)
#@TODO: include GoogleTest find and set test target
#include( CTest )

#gtest_discover_tests(
#        Valuation
#        TEST_LIST   Valuation_targets
#        WORKING_DIRECTORY   test
#)

# set each target to timeout if not finished within 10 sec
#set_tests_properties(${Valuation_targets} PROPERTIES TIMEOUT 0)