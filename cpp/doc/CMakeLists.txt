# LaTeX documentation, built as part of the normal build when pdflatex exists.
#
# nabia_cpp_reference.tex is the reference of this C++ project; the theory notes mirror the
# library layout (doc/forecast/..., doc/optimization/..., doc/regression/...).
# Each PDF is compiled in the matching folder of the build tree, e.g.
#   build/debug/doc/forecast/kalman_filter.pdf
# and then copied next to its source, e.g. doc/forecast/kalman_filter.pdf, so
# the PDFs can be read straight from the repository.
# Build them all with:  cmake --build --preset debug --target nabia_docs
find_package(LATEX COMPONENTS PDFLATEX)
if(NOT LATEX_PDFLATEX_FOUND)
    message(WARNING "pdflatex not found: skipping the documentation (set NABIA_BUILD_DOCS=OFF to silence)")
    return()
endif()

set(NABIA_DOC_STYLE ${CMAKE_CURRENT_SOURCE_DIR}/common/nabiadoc.sty)
add_custom_target(nabia_docs ALL)

# nabia_add_latex_doc(<path relative to doc/>.tex [DESTINATION <dir>] [SNIPPETS <script>])
# Adds target nabia_doc_<name> and attaches it to nabia_docs. pdflatex runs
# three times (labels, then the table of contents, which shifts pages, then the
# final cross-references); its console output goes to <name>.passN.txt and
# errors to <name>.log next to the PDF in the build tree.
#   DESTINATION  where the PDF is copied (default: next to the .tex source).
#   SNIPPETS     a Python script that extracts source excerpts into the build
#                tree before pdflatex runs (it is rerun when any source changes).
function(nabia_add_latex_doc tex)
    cmake_parse_arguments(ARG "" "DESTINATION;SNIPPETS" "" ${ARGN})
    get_filename_component(name ${tex} NAME_WE)
    get_filename_component(subdir ${tex} DIRECTORY)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${tex})
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${subdir})
    set(pdf ${out}/${name}.pdf)
    if(ARG_DESTINATION)
        set(pdf_in_source ${ARG_DESTINATION}/${name}.pdf)
    else()
        set(pdf_in_source ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/${name}.pdf)
    endif()
    get_filename_component(pdf_destination ${pdf_in_source} DIRECTORY)
    set(texinputs "${CMAKE_CURRENT_SOURCE_DIR}/common:")
    set(pre_commands "")
    set(extra_depends "")
    if(ARG_SNIPPETS)
        find_package(Python3 REQUIRED COMPONENTS Interpreter)
        set(snippets ${out}/snippets)
        set(texinputs "${CMAKE_CURRENT_SOURCE_DIR}/common:${snippets}:")
        set(pre_commands COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SNIPPETS} ${PROJECT_SOURCE_DIR} ${snippets})
        file(GLOB_RECURSE extra_depends CONFIGURE_DEPENDS
            ${PROJECT_SOURCE_DIR}/include/*.h ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/*.hpp
            ${PROJECT_SOURCE_DIR}/apps/*.cpp ${PROJECT_SOURCE_DIR}/apps/*.hpp
            ${PROJECT_SOURCE_DIR}/bindings/src/*.cpp ${PROJECT_SOURCE_DIR}/bindings/src/*.hpp
            ${PROJECT_SOURCE_DIR}/bindings/nabia_python_bridge/*.py ${PROJECT_SOURCE_DIR}/bindings/CMakeLists.txt
            ${PROJECT_SOURCE_DIR}/pyproject.toml ${PROJECT_SOURCE_DIR}/scripts/*
            ${PROJECT_SOURCE_DIR}/../.github/workflows/*.yml ${PROJECT_SOURCE_DIR}/../python/scripts/*)
        list(APPEND extra_depends ${CMAKE_CURRENT_SOURCE_DIR}/${ARG_SNIPPETS})
    endif()
    # TEXINPUTS: the shared style in doc/common (and the snippets), then the default path (trailing ':').
    set(latex ${CMAKE_COMMAND} -E env "TEXINPUTS=${texinputs}"
        ${PDFLATEX_COMPILER} -interaction=nonstopmode -halt-on-error -file-line-error
        -output-directory=${out} ${src})
    add_custom_command(
        OUTPUT  ${pdf} ${pdf_in_source}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out}
        ${pre_commands}
        COMMAND ${latex} > ${out}/${name}.pass1.txt
        COMMAND ${latex} > ${out}/${name}.pass2.txt
        COMMAND ${latex} > ${out}/${name}.pass3.txt
        COMMAND ${CMAKE_COMMAND} -E make_directory ${pdf_destination}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${pdf} ${pdf_in_source}
        DEPENDS ${src} ${NABIA_DOC_STYLE} ${extra_depends}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Building ${subdir}/${name}.pdf"
        VERBATIM)
    add_custom_target(nabia_doc_${name} DEPENDS ${pdf} ${pdf_in_source})
    add_dependencies(nabia_docs nabia_doc_${name})
endfunction()

nabia_add_latex_doc(forecast/exponential_smoothing.tex)
nabia_add_latex_doc(forecast/kalman_filter.tex)
nabia_add_latex_doc(forecast/gaussian_process.tex)
nabia_add_latex_doc(optimization/convex/convex_optimization.tex)
nabia_add_latex_doc(regression/elastic_net_regression.tex)

# Tutorial: wheels, cibuildwheel and GitHub Actions (shows the real workflow and pyproject.toml).
nabia_add_latex_doc(packaging/wheels_and_github_actions.tex SNIPPETS packaging/extract_files.py)

# Reference of the C++ project: architecture, methods and code (excerpts extracted from the sources).
nabia_add_latex_doc(nabia_cpp_reference.tex SNIPPETS extract_snippets.py)
