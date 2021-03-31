/*****************************************
@file
@brief Functions to get version and build option strings
- get_kotekan_version
- get_git_branch
- get_git_commit_hash
- get_cmake_build_options
*****************************************/

#ifndef VERSION_H
#define VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Get the kotekan version string
 * The version is generated using versioneer, see python/kotekan/_version.py
 * This version is updated every time make is called
 *
 * @return The kotekan version string
 */
char* get_kotekan_version();

/**
 * @brief Get the git branch string
 *
 * @return char* The git branch string
 */
char* get_git_branch();

/**
 * @brief Get the full git commit hash as a string
 *
 * @return The commit hash string
 */
char* get_git_commit_hash();

/**
 * @brief Get the main kotekan cmake build options as
 *        multiline string
 *
 * @return The cmake options string
 */
char* get_cmake_build_options();

#ifdef __cplusplus
}
#endif

#endif
