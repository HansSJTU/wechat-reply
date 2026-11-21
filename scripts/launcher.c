// Dedicated launcher for the WeChat monitor LaunchAgent.
//
// macOS privacy checks (Documents folder, other apps' data) are charged to the
// "responsible" process. Spawning the monitor as a child -- instead of exec'ing
// it -- keeps this small binary responsible, so the permission is granted to
// this launcher alone rather than to uv/python for every script on the machine.
//
// build: scripts/install.sh
#include <errno.h>
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
    char *argv[] = {"uv", "run", "-q", PROJECT_DIR "/monitor.py", NULL};
    if (posix_spawn(&child, UV_PATH, NULL, NULL, argv, environ) != 0) {
        perror("spawn " UV_PATH);
        return 1;
    }
    signal(SIGTERM, forward);
    signal(SIGINT, forward);
    signal(SIGHUP, forward);
    int status = 0;
    pid_t w;
    while ((w = waitpid(child, &status, 0)) < 0 && errno == EINTR) {
    }
    return (w > 0 && WIFEXITED(status)) ? WEXITSTATUS(status) : 1;
}
