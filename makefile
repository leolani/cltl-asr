SHELL = /bin/bash


SHELL = /bin/bash

project_root ?= $(realpath ..)
project_name ?= $(notdir $(realpath .))
project_version ?= $(shell cat version.txt)

project_repo ?= ${project_root}/cltl-requirements/leolani
project_mirror ?= ${project_root}/cltl-requirements/mirror

project_dependencies ?= $(addprefix $(project_root)/, cltl-combot)

git_remote ?= https://github.com/leolani

chat_bubble_version = 1.5.0
chat_bubble = https://github.com/dmitrizzle/chat-bubble/archive/refs/tags/v$(chat_bubble_version).tar.gz


include util/make/makefile.base.mk
include util/make/makefile.py.base.mk
include util/make/makefile.git.mk
include util/make/makefile.component.mk


clean: py-clean
	rm -rf src/chatui_app/rest/static/chat-bubble

build: src/chatui_app/rest/static/chat-bubble py-install

src/chatui_app/rest/static/chat-bubble:
	$(info Download $(chat_bubble))
	@mkdir src/chatui_app/rest/static/chat-bubble
	@wget -qO- $(chat_bubble) | \
	        tar -xvzf - -C src/chatui_app/rest/static/chat-bubble --strip-components 1 \
	                chat-bubble-$(chat_bubble_version)/component