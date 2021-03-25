library(IRdisplay)

# system2() wrapper for surfacing output in a notebook (since IRKernel is bad at it)
nbsystem2 <- function (command, args = character()) {
    # We'll send the output to a temp file because console output doesn't get pulled through to notebook:
    tmpfile <- tempfile()
    # We want to stop() on failure of the command, not warn()! but preserve existing global settings:
    warnlevel <- options("warn")$warn
    result <- NULL
    tryCatch({
        options(warn=2)
        result <- system2(command, args=args, stdout=tmpfile, stderr=tmpfile)
        display(readLines(tmpfile))
    }, error=function(cond) {
        display(cond)  # Nice to put an error output up top as well as at the bottom, for visibility
        display(readLines(tmpfile))
        stop("Error running system2 command - see logs above for details")
    }, finally={
        file.remove(tmpfile)
        options(warn=warnlevel)
    })
    return(result)
}
