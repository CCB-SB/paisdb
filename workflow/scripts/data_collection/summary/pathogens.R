##################################################
# IMPORTS
##################################################
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(tidyverse))

##################################################
# MAIN
##################################################
palette_6_pastel <- c(
  DO = "#51A0EE", 
  disbiome = "#E34C4E", 
  pathodb = "#6FBB6F",
  wikipedia = "#76549A",
  inhouse = "#CAB2D6",
  gc = "#FF9F40")

#
##  Pathogens
#

pathos = read.csv(snakemake@input['csv'])
pathos %>% head()

# By Source

df = pathos %>% group_by(source) %>% count() %>%
  mutate(perc = round(n/nrow(pathos) * 100, 2)) %>%
  mutate(xtext = "Pathogens") %>%
  arrange(n)
df$label_y = cumsum(df$n)
df$source = factor(df$source, levels = rev(df$source))
patho_by_source <- df %>%
  ggplot(aes(y=n, x = xtext, fill=source)) +
  geom_bar(position='stack', stat='identity') +
  geom_text(aes(y = label_y, label=paste(perc, "%")), vjust=1.5) +
  scale_fill_manual(values=palette_6_pastel)+
  theme_minimal() +
  xlab("")

# By disease_cause
patho_by_cause <- pathos %>% 
  group_by(disease_cause) %>% count() %>%
  mutate(perc = round(n/nrow(pathos) * 100, 2)) %>%
  ggplot(aes(y=n, x=disease_cause)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=paste(perc, "%" )), vjust=1.5, colour = 'white') +
  geom_text(aes(label=n), vjust = -0.5) +
  theme_minimal()

pathos_plots <- plot_grid(
  patho_by_source, patho_by_cause, 
  labels='AUTO')
ggsave2(filename = snakemake@output['png'],
        plot = pathos_plots, device = 'png')