#############################################################
# Files to remove when target clean is called

FILES2CLEAN = $(shell find ./ -name "*.log" \
								-o -name "*.aux" \
								-o -name "*.toc"\
								-o -name "*.out"\
								-o -name "*.lof"\
								-o -name "*.bbl"\
								-o -name "*.blg"\
								-o -name "*.loc"\
								-o -name "*.lot"\
								-o -name "*.idx"\
								-o -name "*.nlo"\
								-o -name "*.ilg"\
								-o -name "*.lol"\
								-o -name "*.bst"\
								-o -name "*.ind"  )


#############################################################
# File to build


THESIS = Hauptdatei

#############################################################
# Command definition

CMD_LATEX 	= pdflatex
CMD_BIB 	= bibtex
CMD_CLEAN 	= rm -rf




#############################################################
# Targets


.PHONY : pdf $(THESIS).pdf

pdf: bib 
	$(CMD_LATEX) $(THESIS)
	$(CMD_LATEX) $(THESIS)
	$(CMD_CLEAN) $(FILES2CLEAN) 


thesis: 
	$(CMD_LATEX) $(THESIS)

bib: thesis
	$(CMD_BIB) $(THESIS)

clean:
	$(CMD_CLEAN) $(FILES2CLEAN) 

clean_all: clean
	$(CMD_CLEAN) $(THESIS).pdf
