// tests the HF cache file removal and blob reclamation on synthetic offline
// fixtures: each case plants a layout (or a hostile variant) and checks what survived

#include "common.h"
#include "download.h"
#include "hf-cache.h"
#include "log.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// the case being checked, printed with every failure
static std::string g_context;

// independent of NDEBUG, so the checks stay alive in Release builds
#define REQUIRE(x) do {                                                         \
    if (!(x)) {                                                                 \
        fprintf(stderr, "%s:%d: [%s] REQUIRE(%s) failed\n",                     \
                __FILE__, __LINE__, g_context.c_str(), #x);                      \
        std::abort();                                                           \
    }                                                                           \
} while (0)

static const std::string COMMIT  = "0123456789abcdef0123456789abcdef01234567";
static const std::string COMMIT2 = "fedcba9876543210fedcba9876543210fedcba98";
static const std::string OID  = "89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567";
static const std::string OID2 = "00abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567";

static fs::path g_root;     // fixture root, outside is the hostile area
static fs::path g_cache;    // directory registered as LLAMA_CACHE
static fs::path g_outside;  // victim files that must always survive

static bool g_symlinks_ok = true;  // probed in main: creating links may need privileges

static void write_file(const fs::path & path, const std::string & content) {
    fs::create_directories(path.parent_path());
    std::ofstream f(path, std::ios::binary);
    f << content;
    REQUIRE(f.good());
}

static void make_link(const fs::path & target, const fs::path & link) {
    std::error_code ec;
    fs::create_directories(link.parent_path(), ec);
    // Windows needs different link calls for files and dirs; relative targets
    // are spelled against the link's dir
    const fs::path resolved = target.is_absolute() ? target : link.parent_path() / target;
    if (fs::is_directory(resolved, ec)) {
        fs::create_directory_symlink(target, link, ec);
    } else {
        fs::create_symlink(target, link, ec);
    }
    REQUIRE(!ec);
}

static bool file_exists(const fs::path & path) {
    std::error_code ec;
    return fs::exists(path, ec);
}

// a planted link is a valid test only if it really names its target
static void require_resolves_to(const fs::path & link, const fs::path & target) {
    std::error_code ec;
    const bool same = fs::equivalent(link, target, ec);
    REQUIRE(!ec);
    REQUIRE(same);
}

// symlink-dependent tests are pointless where links cannot be planted
#define SKIP_WITHOUT_SYMLINKS() do {                                   \
    if (!g_symlinks_ok) {                                              \
        printf("[%s] skipped: symlinks unavailable\n", g_context.c_str()); \
        return;                                                        \
    }                                                                  \
} while (0)

static std::string to_lower_string(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// repo scaffold: models--<id>/refs/main -> COMMIT, with blobs and snapshot dirs
static fs::path make_repo(const std::string & id) {
    const fs::path repo = g_cache / ("models--" + id);
    write_file(repo / "refs" / "main", COMMIT);
    fs::create_directories(repo / "snapshots" / COMMIT);
    fs::create_directories(repo / "blobs");
    return repo;
}

//
// security: nothing outside the repo's blobs dir may be deleted
//

static void test_planted_symlink_targets() {
    g_context = "absolute and relative targets";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("a--victim");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(g_outside / "abs.txt", "data");
    write_file(g_outside / "rel.txt", "data");
    make_link(fs::absolute(g_outside / "abs.txt"), snap / "model-Q4_K_M.gguf");
    make_link("../../../../outside/rel.txt", snap / "evil-Q4_K_M.gguf");
    require_resolves_to(snap / "model-Q4_K_M.gguf", g_outside / "abs.txt");
    require_resolves_to(snap / "evil-Q4_K_M.gguf", g_outside / "rel.txt");

    REQUIRE(common_download_remove("a/victim:Q4_K_M"));
    REQUIRE(file_exists(g_outside / "abs.txt"));
    REQUIRE(file_exists(g_outside / "rel.txt"));

    // a removed symlink pointing outside, with a hash-like basename, must not
    // condemn an unrelated local blob with the same name
    g_context = "external target sharing a name with an unrelated local blob";
    write_file(g_outside / OID, "external");
    write_file(repo / "blobs" / OID, "unrelated");
    make_link(fs::absolute(g_outside / OID), snap / "sharename-Q4_K_M.gguf");
    write_file(repo / "blobs" / OID2, "other");
    make_link("../../blobs/" + OID2, snap / "other-Q4_K_M.gguf");
    require_resolves_to(snap / "sharename-Q4_K_M.gguf", g_outside / OID);
    require_resolves_to(snap / "other-Q4_K_M.gguf", repo / "blobs" / OID2);

    REQUIRE(common_download_remove("a/victim:Q4_K_M"));
    REQUIRE(file_exists(g_outside / OID));
    REQUIRE(file_exists(repo / "blobs" / OID));
    // the real blob of the removed link is still reclaimed
    REQUIRE(!file_exists(repo / "blobs" / OID2));

    g_context = "target through a symlinked directory";
    make_link(g_outside, g_root / "alias");
    write_file(g_outside / "via-dir.txt", "data");
    make_link("../../../../alias/via-dir.txt", snap / "via-dir-Q4_K_M.gguf");
    require_resolves_to(snap / "via-dir-Q4_K_M.gguf", g_outside / "via-dir.txt");

    REQUIRE(common_download_remove("a/victim:Q4_K_M"));
    REQUIRE(file_exists(g_outside / "via-dir.txt"));

    g_context = "target naming the blobs directory";
    make_link("../../blobs/.",  snap / "dot-Q4_K_M.gguf");
    make_link("../../blobs/",   snap / "slash-Q4_K_M.gguf");
    make_link("../../blobs",    snap / "plain-Q4_K_M.gguf");

    REQUIRE(common_download_remove("a/victim:Q4_K_M"));
    REQUIRE(fs::is_directory(repo / "blobs"));

    g_context = "blob kept when its name does not look like an oid";
    write_file(g_outside / "deadbeef.txt", "data");
    make_link("../../../../outside/deadbeef.txt", snap / "hexname-Q4_K_M.gguf");

    REQUIRE(common_download_remove("a/victim:Q4_K_M"));
    REQUIRE(file_exists(g_outside / "deadbeef.txt"));
    // only the planted link itself was removed
    REQUIRE(!file_exists(snap / "hexname-Q4_K_M.gguf"));
}

static void test_symlinked_snapshot_root() {
    g_context = "snapshot root replaced by a symlink to an external dir";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("b--victim");
    const fs::path ext = g_outside / "revdir";
    fs::create_directories(ext);
    write_file(ext / "model-Q4_K_M.gguf", "external model");
    std::error_code ec;
    fs::remove_all(repo / "snapshots" / COMMIT, ec);
    make_link(ext, repo / "snapshots" / COMMIT);
    require_resolves_to(repo / "snapshots" / COMMIT, ext);

    // the entry lives outside the cache, so nothing may be removed
    REQUIRE(!common_download_remove("b/victim:Q4_K_M"));
    REQUIRE(file_exists(ext / "model-Q4_K_M.gguf"));
}

static void test_symlinked_repo_and_blobs() {
    g_context = "repo dir and blobs dir replaced by symlinks";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("c--linkdir");
    const fs::path ext = g_outside / "extdir";
    fs::create_directories(ext / "snapshots" / COMMIT);
    fs::create_directories(ext / "blobs");
    write_file(ext / "blobs" / OID, "external blob");
    make_link("../../blobs/" + OID, ext / "snapshots" / COMMIT / "model-Q4_K_M.gguf");
    std::error_code ec;
    fs::remove_all(repo / "snapshots", ec);
    fs::remove_all(repo / "blobs", ec);
    make_link(ext / "snapshots", repo / "snapshots");
    make_link(ext / "blobs", repo / "blobs");
    require_resolves_to(repo / "snapshots" / COMMIT / "model-Q4_K_M.gguf", ext / "blobs" / OID);
    require_resolves_to(repo / "blobs", ext / "blobs");

    REQUIRE(!common_download_remove("c/linkdir:Q4_K_M"));
    REQUIRE(file_exists(ext / "blobs" / OID));
    REQUIRE(fs::is_symlink(fs::symlink_status(repo / "blobs", ec)));
}

//
// reclamation: what may be deleted, and what must be kept
//

static void test_orphan_reclamation() {
    g_context = "normal orphan reclamation";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("d--normal");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / OID, "gone");
    write_file(repo / "blobs" / OID2, "kept");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    make_link("../../blobs/" + OID2, snap / "model-Q4_K_0.gguf");

    REQUIRE(common_download_remove("d/normal:Q4_K_M"));
    REQUIRE(!file_exists(repo / "blobs" / OID));
    REQUIRE(file_exists(repo / "blobs" / OID2));
    REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
    REQUIRE(file_exists(snap / "model-Q4_K_0.gguf"));
}

static void test_shared_blob_across_revisions() {
    g_context = "blob shared with another revision";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("e--shared");
    const fs::path snap = repo / "snapshots" / COMMIT;
    const fs::path snap2 = repo / "snapshots" / COMMIT2;
    fs::create_directories(snap2);
    write_file(repo / "blobs" / OID, "shared");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    make_link("../../blobs/" + OID, snap2 / "model-Q4_K_M.gguf");

    REQUIRE(common_download_remove("e/shared:Q4_K_M"));
    REQUIRE(file_exists(repo / "blobs" / OID));
    std::error_code ec;
    REQUIRE(fs::is_symlink(fs::symlink_status(snap2 / "model-Q4_K_M.gguf", ec)));

    // the listing still reports only the selected revision: the other
    // revision's entry must not leak into it
    std::string listed_commit;
    const auto files = hf_cache::get_cached_files("e/shared", &listed_commit);
    REQUIRE(listed_commit == COMMIT);
    size_t count = 0;
    for (const auto & f : files) {
        REQUIRE(f.path != "model-Q4_K_M.gguf");
        count++;
    }
    REQUIRE(count == 0);
}

static void test_revision_pinned_between_listing_and_removal() {
    g_context = "ref changes between listing and removal";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("t--tworevs");
    const fs::path snap = repo / "snapshots" / COMMIT;
    const fs::path snap2 = repo / "snapshots" / COMMIT2;
    fs::create_directories(snap2);
    write_file(repo / "blobs" / OID, "blobA");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    write_file(repo / "blobs" / OID2, "blobB");
    make_link("../../blobs/" + OID2, snap2 / "model-Q4_K_M.gguf");

    // list revision A, then make the ref point to revision B before removing
    std::string commit;
    const auto files = hf_cache::get_cached_files("t/tworevs", &commit);
    REQUIRE(commit == COMMIT);
    REQUIRE(files.size() == 1);
    write_file(repo / "refs" / "main", COMMIT2);

    std::vector<std::string> to_remove;
    for (const auto & f : files) {
        to_remove.push_back(f.path);
    }
    REQUIRE(hf_cache::remove_cached_files("t/tworevs", commit, to_remove));

    // revision A lost its entry and blob, revision B is untouched
    REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
    REQUIRE(!file_exists(repo / "blobs" / OID));
    REQUIRE(file_exists(snap2 / "model-Q4_K_M.gguf"));
    REQUIRE(file_exists(repo / "blobs" / OID2));
}

static void test_aliased_references() {
    g_context = "same blob through different spellings";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("f--alias");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / OID, "blob");
    make_link("blobs", repo / "blobs-alias");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    make_link(fs::absolute(repo / "blobs-alias") / OID, snap / "keep-BF16.gguf");
    require_resolves_to(repo / "blobs-alias", repo / "blobs");
    require_resolves_to(snap / "keep-BF16.gguf", repo / "blobs" / OID);

    REQUIRE(common_download_remove("f/alias:Q4_K_M"));
    REQUIRE(file_exists(repo / "blobs" / OID));
}

static void test_aliased_entry_names() {
    g_context = "reference behind an arbitrary entry name";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("r--arbitrary");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / OID, "blob");
    // the removed file links to a sibling whose name is not an oid;
    // that sibling is itself a link to the blob
    make_link("../../blobs/" + OID, snap / "model-Q4_K_0.gguf");
    make_link("model-Q4_K_0.gguf", snap / "model-Q4_K_M.gguf");

    REQUIRE(common_download_remove("r/arbitrary:Q4_K_M"));
    // the blob is still referenced through the sibling, so it must stay
    REQUIRE(file_exists(repo / "blobs" / OID));
    REQUIRE(file_exists(snap / "model-Q4_K_0.gguf"));
}

static void test_case_preserved_blob_names() {
    g_context = "upper-case blob name is removed by its own spelling";
    SKIP_WITHOUT_SYMLINKS();
    const std::string UPPER = "89ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF01234567";
    const std::string lower = to_lower_string(UPPER);
    const fs::path repo = make_repo("s--case");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / UPPER, "upper");
    write_file(repo / "blobs" / lower, "lower");
    make_link("../../blobs/" + UPPER, snap / "model-Q4_K_M.gguf");

    // on a case-insensitive filesystem both spellings name the same file
    std::error_code ec;
    const bool case_insensitive = fs::equivalent(repo / "blobs" / UPPER, repo / "blobs" / lower, ec);

    REQUIRE(common_download_remove("s/case:Q4_K_M"));
    // the blob actually pointed to is gone; the distinct other case stays
    REQUIRE(!file_exists(repo / "blobs" / UPPER));
    if (!case_insensitive) {
        REQUIRE(file_exists(repo / "blobs" / fs::path(lower)));
    }
    REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
}

static void test_blob_symlink_keeps_target() {
    g_context = "blob entry is itself a symlink";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("g--bloblink");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(g_outside / "shared-storage.bin", "shared");
    make_link(fs::absolute(g_outside / "shared-storage.bin"), repo / "blobs" / OID);
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");

    REQUIRE(common_download_remove("g/bloblink:Q4_K_M"));
    // the link entry is gone, the shared storage survived
    REQUIRE(!file_exists(repo / "blobs" / OID));
    REQUIRE(file_exists(g_outside / "shared-storage.bin"));
}

static void test_directory_symlink_blocks_reclaim() {
    g_context = "directory symlink inside a snapshot blocks reclamation";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("v--dirlink");
    const fs::path snap = repo / "snapshots" / COMMIT;
    const fs::path ext = g_outside / "nested";
    fs::create_directories(ext);
    write_file(repo / "blobs" / OID, "blob");
    // a remaining snapshot exposes the blob through a directory symlink the
    // reference scan cannot see into
    make_link(fs::absolute(repo / "blobs" / OID), ext / "model-Q4_K_M.gguf");
    make_link(ext, snap / "subdir");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    require_resolves_to(snap / "subdir" / "model-Q4_K_M.gguf", repo / "blobs" / OID);

    REQUIRE(common_download_remove("v/dirlink:Q4_K_M"));
    // the removed entry is gone, but the still-reachable blob must stay
    REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
    REQUIRE(file_exists(repo / "blobs" / OID));
}

// ".." through a directory symlink goes where the filesystem says, not where
// a lexical path collapse would put it: never collapse ".." away
static void test_dotdot_through_dir_symlink() {
    g_context = "dotdot through a dir symlink must not condemn a local blob";
    SKIP_WITHOUT_SYMLINKS();
    {
        const fs::path repo = make_repo("w--dots");
        const fs::path snap = repo / "snapshots" / COMMIT;
        write_file(repo / "blobs" / OID, "unrelated local blob");
        // escape dir symlink named like a removed model: it goes away with
        // the removal, so it cannot mask the case from the scan
        make_link(fs::absolute(g_root), snap / "esc-Q4_K_M.gguf");
        // fs target: outside the cache and nonexistent; a lexical collapse
        // would instead hit the repo's blobs dir with the same blob name
        make_link("esc-Q4_K_M.gguf/../../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
        require_resolves_to(snap / "esc-Q4_K_M.gguf", g_root);
        // the fs target must dangle: it names blobs/ above the cache root,
        // not the repo's local blob a lexical collapse would hit
        REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));

        REQUIRE(common_download_remove("w/dots:Q4_K_M"));
        REQUIRE(file_exists(repo / "blobs" / OID));
        REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
        REQUIRE(!file_exists(snap / "esc-Q4_K_M.gguf"));
    }

    g_context = "dotdot through a dir symlink still counts as a reference";
    {
        const fs::path repo = make_repo("x--dots");
        const fs::path snap  = repo / "snapshots" / COMMIT;
        const fs::path snap2 = repo / "snapshots" / COMMIT2;
        fs::create_directories(snap2);
        write_file(repo / "blobs" / OID, "blob");
        // the dir symlink sits at repo level, invisible to the snapshot scan;
        // the kept revision references the blob through it and ".."
        make_link(fs::absolute(snap2), repo / "esc");
        make_link("../../esc/../../blobs/" + OID, snap2 / "keep-Q4_K_0.gguf");
        require_resolves_to(repo / "esc", snap2);
        require_resolves_to(snap2 / "keep-Q4_K_0.gguf", repo / "blobs" / OID);
        make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");

        REQUIRE(common_download_remove("x/dots:Q4_K_M"));
        REQUIRE(file_exists(repo / "blobs" / OID));
        REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
        REQUIRE(file_exists(snap2 / "keep-Q4_K_0.gguf"));
    }
}

//
// shapes that must keep working
//

static void test_regular_files_and_nested_paths() {
    g_context = "regular files without blobs (Windows fallback shape)";
    {
        const fs::path repo = make_repo("h--plain");
        const fs::path snap = repo / "snapshots" / COMMIT;
        write_file(snap / "model-Q4_K_M.gguf", "plain");

        REQUIRE(common_download_remove("h/plain:Q4_K_M"));
        REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
    }

    if (!g_symlinks_ok) {
        printf("[%s] remaining blocks skipped: symlinks unavailable\n", g_context.c_str());
        return;
    }

    g_context = "nested model paths";
    {
        const fs::path repo = make_repo("i--nested");
        const fs::path snap = repo / "snapshots" / COMMIT;
        const fs::path quant = snap / "Q4_K_M";
        write_file(repo / "blobs" / OID, "blob");
        make_link("../../../blobs/" + OID, quant / "model-Q4_K_M-00001-of-00002.gguf");
        write_file(quant / "model-Q4_K_M-00002-of-00002.gguf", "plain");

        REQUIRE(common_download_remove("i/nested:Q4_K_M"));
        REQUIRE(!file_exists(quant / "model-Q4_K_M-00001-of-00002.gguf"));
        REQUIRE(!file_exists(quant / "model-Q4_K_M-00002-of-00002.gguf"));
        REQUIRE(!file_exists(repo / "blobs" / OID));
    }

    g_context = "dangling snapshot link";
    {
        const fs::path repo = make_repo("j--dangling");
        const fs::path snap = repo / "snapshots" / COMMIT;
        make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");

        REQUIRE(common_download_remove("j/dangling:Q4_K_M"));
        REQUIRE(!file_exists(snap / "model-Q4_K_M.gguf"));
    }

    g_context = "non-ASCII file name";
    {
        const fs::path repo = make_repo("k--utf8");
        const fs::path snap = repo / "snapshots" / COMMIT;
        write_file(repo / "blobs" / OID, "blob");
        make_link("../../blobs/" + OID, snap / fs::u8path("modèle-é-Q4_K_M.gguf"));

        REQUIRE(common_download_remove("k/utf8:Q4_K_M"));
        REQUIRE(!file_exists(snap / fs::u8path("modèle-é-Q4_K_M.gguf")));
        REQUIRE(!file_exists(repo / "blobs" / OID));
    }
}

//
// invalid inputs and failure handling
//

static void test_invalid_inputs() {
    g_context = "invalid repo id";
    {
        const fs::path repo = make_repo("l--valid");
        const fs::path snap = repo / "snapshots" / COMMIT;
        write_file(snap / "model-Q4_K_M.gguf", "data");

        REQUIRE(!hf_cache::remove_cached_files("l/valid/../evil", COMMIT, {"model-Q4_K_M.gguf"}));
        REQUIRE(!hf_cache::remove_cached_files("l/valid/", COMMIT, {"model-Q4_K_M.gguf"}));
        REQUIRE(!hf_cache::remove_cached_files("", COMMIT, {"model-Q4_K_M.gguf"}));
        REQUIRE(file_exists(snap / "model-Q4_K_M.gguf"));
    }

    g_context = "invalid file names";
    {
        const fs::path repo = make_repo("m--valid");
        const fs::path snap = repo / "snapshots" / COMMIT;
        write_file(snap / "model-Q4_K_M.gguf", "data");
        fs::create_directories(snap / "sub");

        const std::vector<std::string> bad = {
            "", ".", "..", "../../etc/passwd", "/etc/passwd",
            "sub/../..", "sub/../../model-Q4_K_M.gguf",
        };
        for (const auto & name : bad) {
            REQUIRE(!hf_cache::remove_cached_files("m/valid", COMMIT, {name}));
        }
        REQUIRE(file_exists(snap / "model-Q4_K_M.gguf"));
        REQUIRE(fs::is_directory(snap / "sub"));
        REQUIRE(fs::is_directory(repo / "snapshots"));
    }

    g_context = "invalid revision";
    {
        const fs::path repo = make_repo("m--valid");
        const fs::path snap = repo / "snapshots" / COMMIT;
        REQUIRE(!hf_cache::remove_cached_files("m/valid", "", {"model-Q4_K_M.gguf"}));
        REQUIRE(!hf_cache::remove_cached_files("m/valid", "deadbeef", {"model-Q4_K_M.gguf"}));
        REQUIRE(file_exists(snap / "model-Q4_K_M.gguf"));
    }

    g_context = "no cached revision";
    {
        const fs::path repo = g_cache / "models--n--noref";
        fs::create_directories(repo / "snapshots" / COMMIT);
        fs::create_directories(repo / "blobs");
        write_file(repo / "snapshots" / COMMIT / "model-Q4_K_M.gguf", "data");

        // no ref selects a revision, so the listing finds nothing to remove
        REQUIRE(!common_download_remove("n/noref:Q4_K_M"));
        REQUIRE(file_exists(repo / "snapshots" / COMMIT / "model-Q4_K_M.gguf"));
    }

    g_context = "missing files";
    {
        const fs::path repo = make_repo("o--missing");
        REQUIRE(!hf_cache::remove_cached_files("o/missing", COMMIT, {"nope-Q4_K_M.gguf"}));
        REQUIRE(!hf_cache::remove_cached_files("o/missing", COMMIT, {}));
    }
}

static void test_failed_removals() {
    g_context = "failed removal keeps the blob";
#ifndef _WIN32
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("p--locked");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / OID, "blob");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");

    std::error_code ec;
    // readable and searchable, but not writable: listing works, unlink fails
    fs::permissions(snap, fs::perms::owner_read | fs::perms::owner_exec, ec);
    REQUIRE(!ec);

    // the entry cannot be unlinked, so the blob must stay
    REQUIRE(!common_download_remove("p/locked:Q4_K_M"));
    REQUIRE(file_exists(repo / "blobs" / OID));

    fs::permissions(snap, fs::perms::owner_all, ec);
    REQUIRE(!ec);
#endif
}

static void test_unsearchable_parent() {
#ifndef _WIN32
    g_context = "unsearchable snapshot dir blocks removal";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("u--blind");
    const fs::path snap = repo / "snapshots" / COMMIT;
    write_file(repo / "blobs" / OID, "blob");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");

    std::error_code ec;
    // readable but not searchable: entries inside cannot be inspected
    fs::permissions(snap, fs::perms::owner_read, ec);
    REQUIRE(!ec);

    REQUIRE(!hf_cache::remove_cached_files("u/blind", COMMIT, {"model-Q4_K_M.gguf"}));
    // the failed inspection must not turn into permission to delete
    REQUIRE(file_exists(repo / "blobs" / OID));

    fs::permissions(snap, fs::perms::owner_all, ec);
    REQUIRE(!ec);
    // only now can existence be checked inside the dir again
    REQUIRE(file_exists(snap / "model-Q4_K_M.gguf"));
#endif
}

static void test_incomplete_reference_scan() {
#ifndef _WIN32
    g_context = "unreadable revision blocks reclamation";
    SKIP_WITHOUT_SYMLINKS();
    const fs::path repo = make_repo("q--unreadable");
    const fs::path snap = repo / "snapshots" / COMMIT;
    const fs::path snap2 = repo / "snapshots" / COMMIT2;
    fs::create_directories(snap2);
    write_file(repo / "blobs" / OID, "blob");
    make_link("../../blobs/" + OID, snap / "model-Q4_K_M.gguf");
    // an unreferenced blob, but another revision cannot be scanned
    write_file(repo / "blobs" / OID2, "orphan");
    make_link("../../blobs/" + OID2, snap2 / "model-Q4_K_0.gguf");
    std::error_code ec;
    fs::permissions(snap2, fs::perms::none, ec);
    REQUIRE(!ec);

    REQUIRE(common_download_remove("q/unreadable:Q4_K_M"));
    // removal of the entries happened, but no blob may be reclaimed
    REQUIRE(file_exists(repo / "blobs" / OID));
    REQUIRE(file_exists(repo / "blobs" / OID2));

    fs::permissions(snap2, fs::perms::owner_all, ec);
    REQUIRE(!ec);
#endif
}

int main(int argc, char ** argv) {
    // unbuffered, so a crash cannot swallow the reports already printed
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // the negative cases legitimately log warnings, keep the output readable
    common_log_pause(common_log_main());

    const std::string self = argc > 0 ? argv[0] : "";
    const bool relative = argc > 1 && std::string(argv[1]) == "--relative";

    g_root = fs::temp_directory_path() /
             ("test-hf-cache-remove-" + std::to_string(getpid()) + (relative ? "-rel" : ""));
    std::error_code ec;
    fs::remove_all(g_root, ec);
    fs::create_directories(g_root, ec);
    g_outside = g_root / "outside";
    fs::create_directories(g_outside, ec);

    // creating symlinks needs privileges on some systems: probe once, so
    // regular-file cases can still run where link cases cannot
    {
        const fs::path probe = g_root / "probe";
        fs::create_directories(probe, ec);
        write_file(probe / "f", "x");
        fs::create_symlink(probe / "f", probe / "l", ec);
        g_symlinks_ok = !ec;
        fs::remove_all(probe, ec);
        if (!g_symlinks_ok) {
            printf("symlinks unavailable: running regular-file cases only\n");
        }
    }

    if (relative) {
        // relative cache path, resolved from the fixture root
        g_cache = fs::path("rel-cache");
        fs::current_path(g_root);
        fs::create_directories(g_root / g_cache);
        common_set_env("LLAMA_CACHE", "rel-cache");
    } else {
        // absolute path going through a symlink, like a relocated homedir
        const fs::path real = g_root / "cache-real";
        fs::create_directories(real);
        if (g_symlinks_ok) {
            make_link(real, g_root / "cache-link");
            g_cache = g_root / "cache-link";
        } else {
            g_cache = real;
        }
        common_set_env("LLAMA_CACHE", g_cache.string());
    }

    test_planted_symlink_targets();
    test_symlinked_snapshot_root();
    test_symlinked_repo_and_blobs();
    test_orphan_reclamation();
    test_shared_blob_across_revisions();
    test_revision_pinned_between_listing_and_removal();
    test_aliased_references();
    test_aliased_entry_names();
    test_case_preserved_blob_names();
    test_blob_symlink_keeps_target();
    test_directory_symlink_blocks_reclaim();
    test_dotdot_through_dir_symlink();
    test_regular_files_and_nested_paths();
    test_invalid_inputs();
    test_failed_removals();
    test_incomplete_reference_scan();
    test_unsearchable_parent();

    fs::remove_all(g_root, ec);

    // re-run the suite with a relative cache path, covering both spellings
    if (!relative && !self.empty()) {
        const std::string cmd = "\"" + self + "\" --relative";
        REQUIRE(::system(cmd.c_str()) == 0);
    }

    printf("test-hf-cache-remove: all tests OK\n");
    return 0;
}
