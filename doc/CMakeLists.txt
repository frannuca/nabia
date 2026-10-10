# LaTeX documentation, built as part of the normal build when pdflatex exists.
#
# Sources mirror the library layout (doc/forecast/..., doc/optimization/...).
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

# nabia_add_latex_doc(<path relative to doc/>.tex)
# Adds target nabia_doc_<name> and attaches it to nabia_docs. pdflatex runs
# three times (labels, then the table of contents, which shifts pages, then the
# final cross-references); its console output goes to <name>.passN.txt and
# errors to <name>.log next to the PDF.
function(nabia_add_latex_doc tex)
    get_filename_component(name ${tex} NAME_WE)
    get_filename_component(subdir ${tex} DIRECTORY)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/${tex})
    set(out ${CMAKE_CURRENT_BINARY_DIR}/${subdir})
    set(pdf ${out}/${name}.pdf)
    set(pdf_in_source ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/${name}.pdf)
    # TEXINPUTS: the shared style in doc/common, then the default search path (trailing ':').
    set(latex ${CMAKE_COMMAND} -E env "TEXINPUTS=${CMAKE_CURRENT_SOURCE_DIR}/common:"
        ${PDFLATEX_COMPILER} -interaction=nonstopmode -halt-on-error -file-line-error
        -output-directory=${out} ${src})
    add_custom_command(
        OUTPUT  ${pdf} ${pdf_in_source}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${out}
        COMMAND ${latex} > ${out}/${name}.pass1.txt
        COMMAND ${latex} > ${out}/${name}.pass2.txt
        COMMAND ${latex} > ${out}/${name}.pass3.txt
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${pdf} ${pdf_in_source}
        DEPENDS ${src} ${NABIA_DOC_STYLE}
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
