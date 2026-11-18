// Dedicated launcher for the labeling-page LaunchAgent (stt/label.py serve).
//
// Same idea as launcher.c: the project lives in ~/Documents, and macOS charges
// that privacy check to the "responsible" process. Spawning label.py as a child
// keeps this binary responsible, so the Documents grant goes to "WeChat Label"
// alone. Kept separate from launcher.c so changing one never rebuilds (and
// re-signs) the other, which would drop that app's privacy grants.
//
// build: scripts/install.sh label
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PROJECT_DIR
#error "define PROJECT_DIR"
#endif
#ifndef UV_PATH
#error "define UV_PATH"
#endif

extern char **environ;
static pid_t child = 0;

static void forward(int sig) {
    if (child > 0) kill(child, sig);
}

int main(void) {
    if (chdir(PROJECT_DIR) != 0) {
        perror("chdir " PROJECT_DIR);
        return 1;
    }
    char *argv[] = {"uv", "run", "-q", PROJECT_DIR "/stt/label.py", "serve", NULL};
    if (posix_spawn(&child, UV_PATH, NULL, NULL, argv, environ) != 0) {
        perror("spawn " UV_PATH);
        return 1;
    }
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
