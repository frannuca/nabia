# nabia_set_warnings(<target>)
# Enables a strict, but practical, set of warnings on a target.
function(nabia_set_warnings target)
    if(MSVC)
        set(warnings /W4 /permissive-)
        if(NABIA_WARNINGS_AS_ERRORS)
            list(APPEND warnings /WX)
        endif()
    else()
        set(warnings
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wconversion
            -Wsign-conversion
            -Wold-style-cast
            -Wnon-virtual-dtor
            -Woverloaded-virtual
            -Wnull-dereference
            -Wdouble-promotion)
        if(NABIA_WARNINGS_AS_ERRORS)
            list(APPEND warnings -Werror)
        endif()
    endif()
    target_compile_options(${target} PRIVATE ${warnings})
endfunction()
