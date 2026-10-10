# Builds convex_optimization.pdf from LaTeX as part of the normal build.
# Output: <build dir>/src/optimization/convex/docs/convex_optimization.pdf
find_package(LATEX COMPONENTS PDFLATEX)
if(NOT LATEX_PDFLATEX_FOUND)
    message(WARNING "pdflatex not found: skipping the convex optimization docs (set NABIA_BUILD_DOCS=OFF to silence)")
    return()
endif()

set(doc_name convex_optimization)
set(doc_tex  ${CMAKE_CURRENT_SOURCE_DIR}/${doc_name}.tex)
set(doc_pdf  ${CMAKE_CURRENT_BINARY_DIR}/${doc_name}.pdf)
set(pdflatex_cmd
    ${PDFLATEX_COMPILER} -interaction=nonstopmode -halt-on-error -file-line-error
    -output-directory=${CMAKE_CURRENT_BINARY_DIR} ${doc_tex})

# Three passes: labels, then the table of contents (which shifts pages), then
# the final cross-references.
# On failure, see convex_optimization.log next to the PDF.
add_custom_command(
    OUTPUT  ${doc_pdf}
    COMMAND ${pdflatex_cmd} > ${doc_name}.pass1.txt
    COMMAND ${pdflatex_cmd} > ${doc_name}.pass2.txt
    COMMAND ${pdflatex_cmd} > ${doc_name}.pass3.txt
    DEPENDS ${doc_tex}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Building ${doc_name}.pdf with pdflatex"
    VERBATIM)

add_custom_target(nabia_convex_docs ALL DEPENDS ${doc_pdf})
