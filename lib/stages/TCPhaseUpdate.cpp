#include "TCPhaseUpdate.hpp"

#include "StageFactory.hpp"

#define SEC2HR 1/3600.0;
#define Deg2HR 1/15.0;
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif
#define C_SPEED 299792458.0
#define R2D 180. / PI
#define D2R PI / 180.

//#define phase_index(ichan,jbeam,kant,nBeams,nAnts) ((nBeams * ichan + jbeam)*nAnts + kant)

REGISTER_KOTEKAN_STAGE(TCPhaseUpdate);

TCPhaseUpdate::TCPhaseUpdate(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& buffer_container) :
    BeamformingPhaseUpdate(config, unique_name, buffer_container) {}

void TCPhaseUpdate::compute_phases(uint8_t* out_frame, const timespec& gps_time,
                                   const std::vector<freq_id_t>& frequencies_in_frame,
                                   uint8_t* gains_frame) {
    // These lines are just to suppress warnings, remove once function uses them
    //(void)out_frame;
    //(void)gps_time;
    //(void)frequencies_in_frame;
    // Keep this one, since it isn't used for the TC version
    (void)gains_frame;
    
    int ichan, ibeam, iant;
    
    struct tm *timeinfo = localtime(&current_time.tv_sec);
    
    unsigned int hours, minutes, seconds, day, month, year;
    year = timeinfo->tm_year + 1900;
    month = timeinfo->tm_mon + 1;
    day = timeinfo->tm_mday;
    hours =  timeinfo->tm_hour;
    minutes =  timeinfo->tm_min;
    seconds = timeinfo->tm_sec;
    double time_nsec = current_time.tv_nsec/1e9;
    if (month < 3.0){
           month += 12.0;
           year -= 1.0;
       }
    
    
    double JD,AA,BB;
    
    AA = (double)((int)(year/100.0));
    BB = 2 - AA + (double)((int)(AA/4.0));
    JD = BB + (double) ((int)(365.25 * year)) + (double)((int)(30.6001 * (month + 1))) + day + 1720994.5;
    double T = (JD - 2451545.0)/36525.0;
    
    //get gmst in hours
    double gmst =  (24110.54841 + (8640184.812866*T) + (0.093104*T*T - 0.0000063*T*T*T))/3600.;
    gmst = fmod(gmst, 24.);
    double UT = hours + (minutes/60.) + (seconds + time_nsec)/3600.;
    double GSMT = fmod((gmst + UT * 1.002737909), 24.);
    
    //get the local sidereal time in hours
    double LST = GSMT + _inst_long/15.;
        while (LST < 0) {
            LST = LST + 24;
        }
        LST = fmod(LST, 24);
    
  
    for (ichan=0; ichan < _num_local_freq; ichan++){
        double constant = 2*M_PI * frequencies_in_frame[ichan]/C_SPEED;
        for (ibeam=0; ibeam < _num_beams; ibeam++){
            
            /*-----------------------------------------------
             * Coordinates conversion
            *------------------------------------------------*/
            double ha = LST - _beam_coord.ra[ibeam];
            if (ha > 12.0)
                    ha -= 24.0;
                else if (ha < -12.0)
                    ha *=15.0;
            
            //convert angels to radians
            double rha = ha * D2R;
            double rlat= _inst_lat * D2R;
            double rdec = _beam_coord.dec[ibeam] * D2R;
            
            double xhor = cos(rha) * cos(rdec) * sin(rlat) - sin(rdec) * cos(rlat);
            double yhor = sin(rha) * cos(rdec);
            double zhor = cos(rha) * cos(rdec)* cosLat + sin(rdec) * sin(rlat);
                
            //get az/el in degrees
            double az =  atan2f(yhor, xhor) * R2D + 180.0;
                if (az >= 360) az -= 360;
            double el = asinf(zhor)*R2D;
            
            for (iant = 0, iant<feed_locations.size(); iant++){
                double ew_projection = feed_locations[iant].first * cos(el) * sin(az);
                double ns_projection = feed_locations[iant].second * cos(el) * cos(az);
            
                int out_index =(_num_beams * ichan + ibeam) * _num_elements + iant;
                
                double geometric_delay = constant * (ew_projection + ns_projection);
                double real_phases = cos(geometric_delay);
                double imag_phases = -sin(geometric_delay);
                
                //Quantise phases
                
                //out_frame[out_index] =
                
            
                
                
            }
        }
        
    //calculating phases
    
    
    // Code to generate phases goes here.
    // Can access configuration parameters from BeamformingPhaseUpdate
    // e.g. _inst_lat and _inst_long, _beam_coord, _num_beams, etc.
    // Note for this version the `gains_frame` will be set to nullptr
    // and isn't used, since gains are loaded into the GPU separately.
}
