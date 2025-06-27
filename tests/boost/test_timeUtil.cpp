#define BOOST_TEST_MODULE "test_timeUtil"

#include <boost/test/included/unit_test.hpp>
#include <filesystem>
#include <inttypes.h>
#include <vector>
#include <time.h>
#include "fmt.hpp"
#include "timeUtil.hpp"

#define ERA_TOL 2e-12 // <~ 7.5 arcseconds

void check_timespec_equal(const timespec &t1, const timespec &t2) {
    BOOST_CHECK_EQUAL(t1.tv_sec, t2.tv_sec);
    BOOST_CHECK_EQUAL(t1.tv_nsec, t2.tv_nsec);
}

void check_era_close(double era1, double era2) {
    BOOST_CHECK_MESSAGE(fabs(era1-era2) < ERA_TOL,
    fmt::format("|era1 - era2| = |{:.14f} - {:.14f}| = {:.3e} < {:e} failed",
                    era1, era2, fabs(era1-era2), ERA_TOL));
}

class TimeData {
public:
    TimeData(const std::vector<std::string> &tstrs);
    ~TimeData() {};
    
    std::vector<timespec> t_unix;
    std::vector<timespec> t_ut1;
    std::vector<double> dUT1;
    std::vector<double> era;
    std::vector<int64_t> nrot;
    size_t size;
};

bool f_exists(const std::string &name) {
    std::ifstream f(name.c_str());
    return f.good();
}

TimeData::TimeData(const std::vector<std::string> &tstrs) {

    std::string filename = "time_dump.txt";
    std::string script = fmt::format("{:s}/timeUtil.py", TEST_SCRIPT_DIR);

    BOOST_REQUIRE_MESSAGE(f_exists(script),
            fmt::format("critical script {:s}/timeUtil.py was not found",
                TEST_SCRIPT_DIR));

    std::string cmd = fmt::format("python3 {:s} isot", script);
    for(const std::string &tstr : tstrs)
        cmd += fmt::format(" {:s}", tstr);
    cmd += fmt::format(" > {:s}", filename);

    std::cout.flush();
    std::system(cmd.c_str());

    std::ifstream file(filename);
    for(std::string line; std::getline(file, line);) {
        std::stringstream ss(line);
        std::istream_iterator<std::string> begin(ss);
        std::istream_iterator<std::string> end;
        std::vector<std::string> words(begin, end);

        if(words[0] == "UNIX") {
            timespec t = {.tv_sec=std::stol(words[1]), .tv_nsec=std::stol(words[3])};
            t_unix.push_back(t);
        }
        else if(words[0] == "UT1") {
            timespec t = {.tv_sec=std::stol(words[1]), .tv_nsec=std::stol(words[3])};
            t_ut1.push_back(t);
        }
        else if(words[0] == "dUT1") {
            dUT1.push_back(stod(words[1]));
        }
        else if(words[0] == "ERA") {
            era.push_back(stod(words[1]));
            nrot.push_back(stol(words[3]));
        }
    }

    BOOST_REQUIRE(t_unix.size() == t_ut1.size()
                  && t_unix.size() == dUT1.size()
                  && t_unix.size() == era.size()
                  && t_unix.size() == nrot.size());

    size = t_unix.size();
}

std::vector<std::string> default_test_times({
    "1995-01-01T12:00:00.0",
    "2000-01-01T11:58:55.816",
    "2000-01-01T12:00:00.0",
    "2005-01-01T12:00:00.0",
    "2010-01-01T12:00:00.0",
    "2015-01-01T12:00:00.0",
    "2015-06-30T23:59:59.0",
    "2015-06-30T23:59:59.999",
    "2015-06-30T23:59:59.999999",
    "2015-06-30T23:59:59.999999999",
    "2015-06-30T23:59:60.0",
    "2015-07-01T00:00:00.0",
    "2020-01-01T12:00:00.0",
    "2025-01-01T12:00:00.0",
    "2025-12-31T23:59:59.0",
    "2025-12-31T23:59:59.000000001",
    "2025-12-31T23:59:59.999999999",
    "2026-01-01T00:00:00",
    "2026-01-01T00:00:00.000000001",
    "2028-01-01T12:00:00.0",
});

BOOST_AUTO_TEST_CASE(_time_to_ut1) {

    /*
    timespec t_J2000 = {.tv_sec=946'727'935L, .tv_nsec=816'000'000L};
    timespec t_J2000_ut1 = {.tv_sec=2'451'545L * 86400L - 65,
                            .tv_nsec=816'000'000L};

    check_timespec_equal(get_UT1_from_time(t_J2000, 0.0), t_J2000_ut1);
    */

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        check_timespec_equal(get_UT1_from_time(dat.t_unix[i], dat.dUT1[i]),
                             dat.t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_time) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        check_timespec_equal(get_time_from_UT1(dat.t_ut1[i], dat.dUT1[i]),
                             dat.t_unix[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_era) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        int64_t nrot = -1;
        double era = get_ERA_from_UT1(dat.t_ut1[i], &nrot);
        check_era_close(era, dat.era[i]);
        BOOST_CHECK_EQUAL(nrot, dat.nrot[i]);
    }
}

BOOST_AUTO_TEST_CASE(_era_to_ut1) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        check_timespec_equal(get_UT1_from_ERA(dat.nrot[i], dat.era[i]),
                             dat.t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_time_to_era) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        double era = get_ERA_from_time(dat.t_unix[i], dat.dUT1[i]);
        check_era_close(era, dat.era[i]);
    }
}

BOOST_AUTO_TEST_CASE(_time_to_ut1_to_time) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        check_timespec_equal(get_time_from_UT1(
                                get_UT1_from_time(dat.t_unix[i], dat.dUT1[i]),
                                dat.dUT1[i]),
                             dat.t_unix[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_time_to_ut1) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        check_timespec_equal(get_UT1_from_time(
                                get_time_from_UT1(dat.t_ut1[i], dat.dUT1[i]),
                                dat.dUT1[i]),
                             dat.t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_ut1_to_era_to_ut1) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        int64_t nrot = -1;
        double era = get_ERA_from_UT1(dat.t_ut1[i], &nrot);

        check_timespec_equal(get_UT1_from_ERA(nrot, era), dat.t_ut1[i]);
    }
}

BOOST_AUTO_TEST_CASE(_era_to_ut1_to_era) {

    TimeData dat(default_test_times);

    BOOST_REQUIRE(dat.size == default_test_times.size());

    for(size_t i = 0; i < dat.size; i++) {
        timespec ut1 = get_UT1_from_ERA(dat.nrot[i], dat.era[i]);

        int64_t nrot = -1;
        double era = get_ERA_from_UT1(ut1, &nrot);

        check_era_close(era, dat.era[i]);
        BOOST_CHECK_EQUAL(nrot, dat.nrot[i]);
    }
}
