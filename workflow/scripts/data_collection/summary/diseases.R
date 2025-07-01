##################################################
# IMPORTS
##################################################
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(cowplot))
suppressPackageStartupMessages(library(tidyverse))
suppressPackageStartupMessages(library("ggVennDiagram"))

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
##  Diseases
#

dis = read.csv(snakemake@input['csv'], na.strings = "")
do = read.csv(snakemake@input['do_terms'], na.strings = "")

# By Source
df = dis %>% group_by(source) %>% count() %>%
  mutate(perc = round(n/nrow(dis) * 100, 2)) %>%
  mutate(xtext = "Diseases") %>%
  arrange(n)
df$label_y = cumsum(df$n)
df$source = factor(df$source, levels = rev(df$source))
dis_by_source <- df %>%
  ggplot(aes(y=n, x = xtext, fill=source)) +
  geom_bar(position='stack', stat='identity') +
  geom_text(aes(y = label_y, label=paste(perc, "%")), vjust=1.5) +
  scale_fill_manual(values=palette_6_pastel)+
  theme_minimal() +
  xlab("")

# By DO
dis_by_do <- dis %>% 
  mutate(DO_entry = if_else(is.na(disease_id), FALSE, TRUE)) %>%
  group_by(DO_entry) %>% count() %>%
  mutate(perc = round(n/nrow(dis) * 100, 2)) %>%
  ggplot(aes(y=n, x=DO_entry)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=paste(perc, "%" )), vjust=1.5, colour = 'white') +
  geom_text(aes(label=n), vjust = -0.5) +
  theme_minimal()

# By caused pathogen
dis_by_caused <- dis %>% 
  group_by(caused_by_pathogen) %>% count() %>%
  mutate(perc = round(n/nrow(dis) * 100, 2)) %>%
  ggplot(aes(y=n, x=caused_by_pathogen)) +
  geom_bar(stat='identity') +
  geom_text(aes(label=paste(perc, "%" )), vjust=1.5, colour = 'white') +
  geom_text(aes(label=n), vjust = -0.5) +
  theme_minimal()

# Overlap with DO terms
set.seed(123)
do_overlap <- ggVennDiagram(list(Diseases=dis$disease,
                   DO_terms=do$name)) + 
  scale_fill_gradient(low="grey90",high = "red")

dis_plots <- plot_grid(dis_by_source, dis_by_do, 
                       dis_by_caused, do_overlap,
          labels='AUTO')

# Save
ggsave2(filename = snakemake@output['png'],
        plot = dis_plots, device = 'png')