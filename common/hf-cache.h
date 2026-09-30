#pragma once

#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

// List files of the cached repos (optionally a single repo). When a repo is
// given, commit receives its cached revision, for use with remove_cached_files.
hf_files get_cached_files(const std::string & repo_id = {}, std::string * commit = nullptr);

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const hf_file & file);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

// Remove files (repo-relative paths, as listed by get_cached_files) from the
// given revision, then reclaim blobs no remaining revision uses.
bool remove_cached_files(const std::string & repo_id, const std::string & commit, const std::vector<std::string> & files);

// Returns the HuggingFace hub cache path
std::string get_cache_path();

} // namespace hf_cache
