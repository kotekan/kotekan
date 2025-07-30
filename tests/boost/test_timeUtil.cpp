#define BOOST_TEST_MODULE "test_timeUtil"

#include "timeUtil.hpp"

#include "fmt.hpp"

#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <inttypes.h>
#include <sys/wait.h>
#include <time.h>
#include <vector>

#define ERA_TOL 2e-12 // <~ 0.5 nanoseconds of rotation
#define NS_TOL 0

void check_timespec_equal(const timespec& t1, const timespec& t2) {
    BOOST_CHECK_EQUAL(t1.tv_sec, t2.tv_sec);

    long ns1 = (long)t1.tv_nsec;
    long ns2 = (long)t2.tv_nsec;
    BOOST_CHECK(abs(ns1-ns2) <= NS_TOL);
}

void check_ns_equal(int64_t t1, int64_t t2) {
    BOOST_CHECK_MESSAGE(abs(t1-t2) <= NS_TOL,
                        fmt::format("|ns1 - ns2| = |{:d} - {:d}| = {:d} < {:d} failed", t1, t2, abs(t1-t2), NS_TOL));
}

void check_era_close(double era1, double era2) {
    BOOST_CHECK_MESSAGE(fabs(era1 - era2) < ERA_TOL,
                        fmt::format("|era1 - era2| = |{:.14f} - {:.14f}| = {:.3e} < {:e} failed",
                                    era1, era2, fabs(era1 - era2), ERA_TOL));
}

bool f_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

static std::vector<std::string> test_times({
    "1995-01-01T12:00:00.0",         "2000-01-01T11:58:55.816",       "2000-01-01T12:00:00.0",
    "2005-01-01T12:00:00.0",         "2010-01-01T12:00:00.0",         "2015-01-01T12:00:00.0",
    "2015-06-30T23:59:59.0",         "2015-06-30T23:59:59.999",       "2015-06-30T23:59:59.999999",
    "2015-06-30T23:59:59.999999999", "2015-06-30T23:59:60.0",         "2015-07-01T00:00:00.0",
    "2020-01-01T12:00:00.0",         "2025-01-01T12:00:00.0",         "2025-12-31T23:59:59.0",
    "2025-12-31T23:59:59.000000001", "2025-12-31T23:59:59.999999999", "2026-01-01T00:00:00",
    "2026-01-01T00:00:00.000000001", "2027-03-14T02:07:01.828123456", "2028-01-01T12:00:00.0",
});

class TimeData {
public:
    TimeData(){BOOST_TEST_MESSAGE("TimeData constructor");};
    void setup();
    ~TimeData(){BOOST_TEST_MESSAGE("TimeData destructor");};

    static std::vector<timespec> t_unix;
    static std::vector<int64_t> t_ut1;
    static std::vector<double> dUT1;
    static std::vector<double> era;
    static std::vector<int64_t> nrot;
    static size_t size;
};

std::vector<timespec> TimeData::t_unix{};
std::vector<int64_t> TimeData::t_ut1{};
std::vector<double> TimeData::dUT1{};
std::vector<double> TimeData::era{};
std::vector<int64_t> TimeData::nrot{};
size_t TimeData::size = 0;

void TimeData::setup() {

    BOOST_TEST_MESSAGE("TimeData setup.");

    const std::vector<std::string>& tstrs = test_times;

    std::string filename = "time_dump.txt";
    std::string script = fmt::format("{:s}/timeUtil.py", TEST_SCRIPT_DIR);

    BOOST_REQUIRE_MESSAGE(
        f_exists(script),
        fmt::format("critical script {:s}/timeUtil.py was not found", TEST_SCRIPT_DIR));

    std::string cmd = fmt::format("python3 {:s} isot", script);
    for (const std::string& tstr : tstrs)
        cmd += fmt::format(" {:s}", tstr);
    cmd += fmt::format(" > {:s}", filename);

    std::cout.flush();
    int sys_ret_val = std::system(cmd.c_str());
    BOOST_REQUIRE_MESSAGE(WEXITSTATUS(sys_ret_val) == 0,
                          fmt::format("system python call terminated abnormally with status {:d}",
                                      WEXITSTATUS(sys_ret_val)));

    std::ifstream file(filename);
    for (std::string line; std::getline(file, line);) {
        std::stringstream ss(line);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> words(begin, end);

        if (words[0] == "UNIX") {
            timespec t = {.tv_sec = std::stol(words[1]), .tv_nsec = std::stol(words[3])};
            t_unix.push_back(t);
        } else if (words[0] == "UT1") {
            t_ut1.push_back(std::stol(words[1]));
        } else if (words[0] == "dUT1") {
            dUT1.push_back(std::stod(words[1]));
        } else if (words[0] == "ERA4") {
            era.push_back(std::stod(words[1]));
            nrot.push_back(std::stol(words[3]));
        }
    }

    BOOST_REQUIRE(t_unix.size() == test_times.size());
    BOOST_REQUIRE(t_unix.size() == t_ut1.size() && t_unix.size() == dUT1.size()
                  && t_unix.size() == era.size() && t_unix.size() == nrot.size());

    size = t_unix.size();

    BOOST_TEST_MESSAGE(fmt::format("Setup TimeData with {:d} entries.", size));
}

BOOST_TEST_GLOBAL_FIXTURE(TimeData);

BOOST_AUTO_TEST_CASE(_time_to_ut1) {

    BOOST_TEST_MESSAGE(fmt::format("Testing time->ut1 with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        check_ns_equal(get_UT1_from_time(TimeData::t_unix[i],
                                         TimeData::dUT1[i]),
                       TimeData::t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_time) {

    BOOST_TEST_MESSAGE(fmt::format("Testing ut1->time with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        check_timespec_equal(get_time_from_UT1(TimeData::t_ut1[i], TimeData::dUT1[i]),
                             TimeData::t_unix[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_era) {

    BOOST_TEST_MESSAGE(fmt::format("Testing ut1->ERA with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        int64_t nrot = -1;
        double era = get_ERA_from_UT1(TimeData::t_ut1[i], &nrot);
        check_era_close(era, TimeData::era[i]);
        BOOST_CHECK_EQUAL(nrot, TimeData::nrot[i]);
    }
}

BOOST_AUTO_TEST_CASE(_era_to_ut1) {

    BOOST_TEST_MESSAGE(fmt::format("Testing era->ut1 with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        check_ns_equal(get_UT1_from_ERA(TimeData::nrot[i], TimeData::era[i]),
                       TimeData::t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_time_to_era) {

    BOOST_TEST_MESSAGE(fmt::format("Testing time->era with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        double era = get_ERA_from_time(TimeData::t_unix[i], TimeData::dUT1[i]);
        check_era_close(era, TimeData::era[i]);
    }
}

BOOST_AUTO_TEST_CASE(_time_to_ut1_to_time) {

    BOOST_TEST_MESSAGE(fmt::format("Testing time->ut1->time with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        check_timespec_equal(
            get_time_from_UT1(get_UT1_from_time(TimeData::t_unix[i], TimeData::dUT1[i]), TimeData::dUT1[i]),
            TimeData::t_unix[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_time_to_ut1) {

    BOOST_TEST_MESSAGE(fmt::format("Testing ut1->time->ut1 with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        check_ns_equal(
            get_UT1_from_time(get_time_from_UT1(TimeData::t_ut1[i],
                                                TimeData::dUT1[i]),
                              TimeData::dUT1[i]),
            TimeData::t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_era_to_ut1) {

    BOOST_TEST_MESSAGE(fmt::format("Testing ut1->era->ut1 with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        int64_t nrot = -1;
        double era = get_ERA_from_UT1(TimeData::t_ut1[i], &nrot);

        check_ns_equal(get_UT1_from_ERA(nrot, era), TimeData::t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_era_to_ut1_to_era) {

    BOOST_TEST_MESSAGE(fmt::format("Testing era->ut1->era with {:d} entries.", TimeData::size));

    for (size_t i = 0; i < TimeData::size; i++) {
        int64_t ut1 = get_UT1_from_ERA(TimeData::nrot[i], TimeData::era[i]);

        int64_t nrot = -1;
        double era = get_ERA_from_UT1(ut1, &nrot);

        check_era_close(era, TimeData::era[i]);
        BOOST_CHECK_EQUAL(nrot, TimeData::nrot[i]);
    }
}
