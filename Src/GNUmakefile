default:
	bison --no-lines --defines=amrexpr_parser.tab.h --output=amrexpr_parser.tab.nolint.H amrexpr_parser.y
	flex --noline --header-file=amrexpr_parser.lex.h --outfile=amrexpr_parser.lex.nolint.H amrexpr_parser.l
